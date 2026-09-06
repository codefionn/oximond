//! Reference evaluator for [`TraceIr`].
//!
//! It runs a trace over a symbolic register file. On a guard failure it
//! returns the state the snapshot restores, so tests can check recovery
//! semantics without native code. Effects are symbolic: each commit appends
//! to a log, and the evaluator never replays one.

use super::ir::{
    BindingKind, FrameId, GuardId, LoopParamInit, META_FIELD_KEY, TERMINAL_GUARD, TableBinding,
    TraceIr, TraceOp, TraceScalar, TraceTerminator, TraceType, ValueId,
};
use super::semantics;
use super::snapshot::{RegisterEntry, Snapshot, VirtualAlloc, VirtualId, VirtualKey};

/// One committed effect.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectRecord {
    pub effect: u32,
    pub label: u32,
    pub operands: Vec<TraceScalar>,
}

/// One outer frame of a multi-frame restore: a caller stopped at its call.
#[derive(Clone, Debug, PartialEq)]
pub struct RestoredFrame {
    pub frame: FrameId,
    pub last_interpreted_pc: Option<usize>,
    pub registers: Vec<TraceScalar>,
}

/// The lane id a virtual table's register holds in a [`Restored`], offset by
/// the table's [`VirtualId`]. Registers that name one virtual table hold one
/// lane, so aliasing shows as equal scalars.
pub const VIRTUAL_LANE_BASE: u64 = 1 << 62;
/// The lane of the string a `Slice` entry names: this base plus the id of its
/// recipe. The reference does not make the substring.
pub const SLICE_LANE_BASE: u64 = 1 << 61;

/// A virtual table as an exit builds it, with every value resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct RestoredTable {
    pub alloc: VirtualAlloc,
    /// The values at keys 1, 2, ... A nil is a hole.
    pub array: Vec<TraceScalar>,
    /// The other fields in recipe order. A nil is a deleted field.
    pub fields: Vec<(VirtualKey, TraceScalar)>,
    /// The table lane the exit sets as the metatable, if the recipe has one.
    pub metatable: Option<TraceScalar>,
}

/// Interpreter state after a restore.
#[derive(Clone, Debug, PartialEq)]
pub struct Restored {
    pub pc: usize,
    pub last_interpreted_pc: Option<usize>,
    /// The innermost frame's registers.
    pub registers: Vec<TraceScalar>,
    pub effect_position: u32,
    /// The frames outside the innermost one, outermost first. Empty unless
    /// the snapshot sits inside an inlined call.
    pub outer: Vec<RestoredFrame>,
    /// The virtual tables some register names, by id, each built once.
    pub tables: Vec<(VirtualId, RestoredTable)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// An entry type check failed. Nothing ran.
    EntryMismatch,
    /// A guard failed.
    GuardFailed { guard: GuardId, restored: Restored },
    /// The trace ended through an exit terminator.
    Exited(Restored),
    /// The iteration limit was reached before the trace left.
    IterationLimit { iterations: u32 },
    /// A function trace returned this value (see
    /// [`super::ir::TraceTerminator::Return`]).
    Returned(Option<TraceScalar>),
}

#[derive(Debug)]
pub struct Evaluation {
    pub outcome: Outcome,
    pub effects: Vec<EffectRecord>,
    pub iterations: u32,
}

/// Forces the guard at one guest location to fail, whatever its operand is.
/// It lets a test take every exit of a trace, including exits the data would
/// never reach, and compare the restored state of two equivalent traces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForcedExit {
    pub frame: FrameId,
    pub pc: usize,
    /// Fails only this guard when set, so each of several guards at one pc
    /// can be forced in turn. Unset fails the first guard evaluated at `pc`.
    pub guard: Option<GuardId>,
    /// The guard fails when this many loop-backs have completed.
    pub iteration: u32,
}

/// The `label` of the [`EffectRecord`] a table store leaves in the log.
pub const TABLE_STORE_LABEL: u32 = u32::MAX - 1;

/// The `label` of the [`EffectRecord`] a helper that writes leaves in the
/// log is this plus its [`HelperId`](super::helper::HelperId).
pub const HELPER_EFFECT_LABEL_BASE: u32 = 0x4000_0000;

/// The table storage the bindings of a trace resolve, as a model. Every
/// method takes the index of a binding of [`TraceIr::bindings`].
pub trait TraceHeap {
    /// Whether entry accepts the model's tables: every binding resolves, a
    /// field holds a value of its bound type, and so on. A refusal is an
    /// entry mismatch.
    fn entry_accepts(&mut self, bindings: &[TableBinding]) -> bool;
    /// The value at array key `key`, or `None` when the slot is outside the
    /// array or holds nil, which fails the guard.
    fn array_get(&mut self, binding: u32, key: TraceScalar) -> Option<TraceScalar>;
    /// Overwrites array slot `key` with `value`. Returns whether the slot
    /// exists with a value of the same type, which the guard needs.
    fn array_set(&mut self, binding: u32, key: TraceScalar, value: TraceScalar) -> bool;
    /// The value of string key `key` (a lane string or a borrow) in the hash
    /// part of the bound table, which has no metatable: nil for a missing
    /// key. `None` when the value is not a scalar (nil, boolean, integer,
    /// float), which fails the guard.
    fn hash_get(&mut self, binding: u32, key: TraceScalar) -> Option<TraceScalar> {
        let _ = (binding, key);
        panic!("the heap model has no hash lookups")
    }
    /// The one-based index of the first slot with a value after slot `key`
    /// of the bound array ([`TraceOp::ArrayNext`]), or of the first slot of
    /// the array when `start`. `None` when no slot follows, or `key` is not
    /// an integral number from 1 to the array's length naming a slot that
    /// holds a value, which fails the guard.
    fn array_next(&mut self, binding: u32, key: TraceScalar, start: bool) -> Option<i64> {
        let _ = (binding, key, start);
        panic!("the heap model has no array steps")
    }
    /// Whether array key `key` names a slot inside the array that holds nil,
    /// which a nil read needs.
    fn array_hole(&mut self, binding: u32, key: TraceScalar) -> bool {
        let _ = (binding, key);
        false
    }
    /// Whether array key `key` lies outside the array part of a solid table
    /// (its hash part is empty), which a nil read past the end needs
    /// ([`TraceOp::ArrayLoad`] with `past_end`).
    fn array_miss(&mut self, binding: u32, key: TraceScalar) -> bool {
        let _ = (binding, key);
        false
    }
    /// Sets key `key` of the bound table to `value` as a raw store,
    /// growing the table as needed ([`TraceOp::ArraySet`]). Returns whether
    /// it stored, which the guard needs.
    fn table_set(&mut self, binding: u32, key: TraceScalar, value: TraceScalar) -> bool {
        let _ = (binding, key, value);
        panic!("the heap model has no table sets")
    }
    /// The result of calling `function` on `args` ([`TraceOp::HostCall`]), or
    /// `None` when the VM cannot answer, which fails the guard.
    fn host_call(
        &mut self,
        function: super::ir::HostFunction,
        args: &[TraceScalar],
    ) -> Option<TraceScalar> {
        let _ = (function, args);
        None
    }
    /// The results of calling the helper `helper` on `args`
    /// ([`TraceOp::CallHelper`]), one per entry of its signature, or `None`
    /// when it misses, which fails the guard.
    fn helper_call(
        &mut self,
        helper: super::helper::HelperId,
        args: &[TraceScalar],
    ) -> Option<Vec<TraceScalar>> {
        let _ = (helper, args);
        None
    }
    /// The bound field's value.
    fn field_get(&mut self, binding: u32) -> TraceScalar;
    /// Overwrites the bound field.
    fn field_set(&mut self, binding: u32, value: TraceScalar);
    /// The bound table's length.
    fn length(&mut self, binding: u32) -> TraceScalar;
    /// Field `field` of [`TraceIr::fields`] of the table `table` (a lane
    /// id), or `None` when the table has no such field with that type,
    /// which fails the guard.
    fn table_field_get(&mut self, table: u64, field: u32) -> Option<TraceScalar> {
        let _ = (table, field);
        panic!("the heap model has no run-time tables")
    }
    /// The value at array key `key` of the table `table` (a lane id), met by
    /// lane ([`TraceOp::LaneArrayLoad`]), or `None` when the slot is outside
    /// the array or holds nil, or the row cannot name the table, which fails
    /// the guard.
    fn lane_array_get(&mut self, table: u64, key: TraceScalar) -> Option<TraceScalar> {
        let _ = (table, key);
        panic!("the heap model has no run-time arrays")
    }
    /// Overwrites array key `key` of the table `table` (a lane id), met by
    /// lane ([`TraceOp::LaneArrayStore`]). Returns whether the slot is inside
    /// the array and holds a non-nil value of the value's type, which the
    /// guard needs; a store that fails it changes nothing.
    fn lane_array_set(&mut self, table: u64, key: TraceScalar, value: TraceScalar) -> bool {
        let _ = (table, key, value);
        panic!("the heap model has no run-time arrays")
    }
    /// Byte `key` (counting from 1) of the bound string, as the number type of
    /// its binding, or `None` when `key` is no integral number inside the
    /// string, which fails the guard.
    fn string_byte(&mut self, binding: u32, key: TraceScalar) -> Option<TraceScalar> {
        let _ = (binding, key);
        panic!("the heap model has no strings")
    }
    /// The byte count of the bound string.
    fn string_length(&mut self, binding: u32) -> TraceScalar {
        let _ = binding;
        panic!("the heap model has no strings")
    }
    /// Overwrites field `field` of table `table`. Returns whether the table
    /// has the field with the value's type, which the guard needs.
    fn table_field_set(&mut self, table: u64, field: u32, value: TraceScalar) -> bool {
        let _ = (table, field, value);
        panic!("the heap model has no run-time tables")
    }
}

/// A heap for traces with no table operation. Any access panics.
struct NoHeap;

impl TraceHeap for NoHeap {
    fn entry_accepts(&mut self, bindings: &[TableBinding]) -> bool {
        assert!(bindings.is_empty(), "the trace needs a heap model");
        true
    }
    fn array_get(&mut self, _: u32, _: TraceScalar) -> Option<TraceScalar> {
        panic!("the trace needs a heap model")
    }
    fn array_set(&mut self, _: u32, _: TraceScalar, _: TraceScalar) -> bool {
        panic!("the trace needs a heap model")
    }
    fn field_get(&mut self, _: u32) -> TraceScalar {
        panic!("the trace needs a heap model")
    }
    fn field_set(&mut self, _: u32, _: TraceScalar) {
        panic!("the trace needs a heap model")
    }
    fn length(&mut self, _: u32) -> TraceScalar {
        panic!("the trace needs a heap model")
    }
}

/// Evaluates `ir` over `registers`, the root frame's interpreter registers.
/// `effect_result` supplies the value of each effect that produces one.
/// `roots` backs `Root` register entries.
pub fn evaluate(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    max_iterations: u32,
    effect_result: impl FnMut(&EffectRecord, u32) -> Option<TraceScalar>,
) -> Evaluation {
    evaluate_forced(ir, registers, roots, max_iterations, None, effect_result)
}

/// [`evaluate`] with an optional forced guard failure.
pub fn evaluate_forced(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    max_iterations: u32,
    forced: Option<ForcedExit>,
    effect_result: impl FnMut(&EffectRecord, u32) -> Option<TraceScalar>,
) -> Evaluation {
    evaluate_with_heap(
        ir,
        registers,
        roots,
        max_iterations,
        forced,
        &mut NoHeap,
        effect_result,
    )
}

/// [`evaluate_forced`] for a trace with table operations, over a model of
/// the tables its bindings resolve.
pub fn evaluate_with_heap(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    max_iterations: u32,
    forced: Option<ForcedExit>,
    heap: &mut dyn TraceHeap,
    mut effect_result: impl FnMut(&EffectRecord, u32) -> Option<TraceScalar>,
) -> Evaluation {
    evaluate_inner(
        ir,
        registers,
        roots,
        max_iterations,
        forced,
        heap,
        &mut effect_result,
    )
}

type EffectResult<'a> = dyn FnMut(&EffectRecord, u32) -> Option<TraceScalar> + 'a;

fn evaluate_inner(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    max_iterations: u32,
    forced: Option<ForcedExit>,
    heap: &mut dyn TraceHeap,
    effect_result: &mut EffectResult<'_>,
) -> Evaluation {
    let mut values: Vec<Option<TraceScalar>> = vec![None; ir.values.len()];
    let mut effects = Vec::new();
    if !heap.entry_accepts(&ir.bindings) {
        return Evaluation {
            outcome: Outcome::EntryMismatch,
            effects,
            iterations: 0,
        };
    }
    for input in &ir.entry.inputs {
        let scalar = registers[usize::from(input.register)];
        if scalar.ty() != input.ty {
            return Evaluation {
                outcome: Outcome::EntryMismatch,
                effects,
                iterations: 0,
            };
        }
        values[input.value.index()] = Some(scalar);
    }
    for param in &ir.loop_params {
        values[param.value.index()] = Some(match param.init {
            LoopParamInit::Input(value) => get(&values, value),
            LoopParamInit::Const(scalar) => scalar,
        });
    }
    let mut iterations = 0_u32;
    // The preheader runs once. Its values stay defined across iterations.
    for instruction in &ir.preheader {
        let outcome = step(
            ir,
            registers,
            roots,
            &mut values,
            &mut effects,
            effect_result,
            heap,
            instruction,
            forced.filter(|forced| forced.iteration == 0),
        );
        if let Some((guard, restored)) = outcome {
            return Evaluation {
                outcome: Outcome::GuardFailed { guard, restored },
                effects,
                iterations,
            };
        }
    }
    loop {
        // Body values are redefined each iteration. Entry inputs and loop
        // parameters stay as assigned.
        for instruction in &ir.body {
            if let Some(result) = instruction.result
                && !matches!(instruction.op, TraceOp::HelperResult { .. })
            {
                values[result.index()] = None;
            }
            let outcome = step(
                ir,
                registers,
                roots,
                &mut values,
                &mut effects,
                effect_result,
                heap,
                instruction,
                forced.filter(|forced| forced.iteration == iterations),
            );
            if let Some((guard, restored)) = outcome {
                return Evaluation {
                    outcome: Outcome::GuardFailed { guard, restored },
                    effects,
                    iterations,
                };
            }
        }
        match &ir.terminator {
            TraceTerminator::Return { value } => {
                return Evaluation {
                    outcome: Outcome::Returned(value.map(|value| get(&values, value))),
                    effects,
                    iterations,
                };
            }
            // The frame starts over with the arguments, one tail call more.
            TraceTerminator::TailCall { args, .. } => {
                let guest = usize::from(ir.frames[0].register_count) - ir.external.len();
                let mut window = registers.to_vec();
                for (register, slot) in window.iter_mut().enumerate().take(guest) {
                    *slot = args
                        .get(register)
                        .map_or(TraceScalar::Nil, |arg| get(&values, *arg));
                }
                if let Some(position) = ir
                    .external
                    .iter()
                    .position(|slot| *slot == super::ir::ExternalSlot::TailCalls)
                    && let TraceScalar::Integer(count) = window[guest + position]
                {
                    window[guest + position] = TraceScalar::Integer(count + 1);
                }
                let mut next = evaluate_inner(ir, &window, roots, 1, None, heap, effect_result);
                let mut all = effects;
                all.append(&mut next.effects);
                next.effects = all;
                return next;
            }
            TraceTerminator::Exit { snapshot }
            | TraceTerminator::Link { snapshot }
            | TraceTerminator::LinkLoop { snapshot, .. } => {
                let restored = restore(&ir.snapshots[snapshot.index()], &values, registers, roots);
                return Evaluation {
                    outcome: Outcome::Exited(restored),
                    effects,
                    iterations,
                };
            }
            TraceTerminator::LoopBack { args } => {
                iterations += 1;
                if iterations >= max_iterations {
                    return Evaluation {
                        outcome: Outcome::IterationLimit { iterations },
                        effects,
                        iterations,
                    };
                }
                // Parallel assignment: read every argument first.
                let next: Vec<TraceScalar> = args.iter().map(|arg| get(&values, *arg)).collect();
                for (param, scalar) in ir.loop_params.iter().zip(next) {
                    values[param.value.index()] = Some(scalar);
                }
            }
        }
    }
}

fn get(values: &[Option<TraceScalar>], value: ValueId) -> TraceScalar {
    values[value.index()].expect("verified trace defines every operand")
}

thread_local! {
    /// Whether the slot each number load read held an integer.
    static NUMBER_TAGS: std::cell::RefCell<std::collections::HashMap<ValueId, bool>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn step(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    values: &mut [Option<TraceScalar>],
    effects: &mut Vec<EffectRecord>,
    effect_result: &mut EffectResult<'_>,
    heap: &mut dyn TraceHeap,
    instruction: &super::ir::TraceInstruction,
    forced: Option<ForcedExit>,
) -> Option<(GuardId, Restored)> {
    let forced_at = |guard: GuardId| {
        forced.is_some_and(|forced| {
            forced.frame == instruction.site.frame
                && forced.pc == instruction.site.pc
                && forced.guard.is_none_or(|only| only == guard)
        })
    };
    let produced = match &instruction.op {
        TraceOp::Constant(scalar) => Some(*scalar),
        // A charge to the collector, which the evaluator does not model.
        TraceOp::PhantomTable { .. } => None,
        // A nested activation: the same trace over a fresh frame holding the
        // arguments, nil, and the caller's cells and globals. A guard that
        // fails inside it surfaces as that guard's own restore: the model
        // does not rebuild the frames of the activations around it.
        TraceOp::SelfCall { args, .. } => {
            let function = ir
                .function
                .as_ref()
                .expect("a self call in a function trace");
            let guest = usize::from(ir.frames[0].register_count) - ir.external.len();
            let mut window = registers.to_vec();
            for (register, slot) in window.iter_mut().enumerate().take(guest) {
                *slot = args
                    .get(register)
                    .filter(|_| register < function.parameters.len())
                    .map_or(TraceScalar::Nil, |arg| get(values, *arg));
            }
            let nested = evaluate_inner(ir, &window, roots, 1, None, heap, effect_result);
            effects.extend(nested.effects);
            match nested.outcome {
                Outcome::Returned(value) => value,
                Outcome::GuardFailed { guard, restored } => return Some((guard, restored)),
                other => panic!("a nested activation ended with {other:?}"),
            }
        }
        TraceOp::Move { source } => Some(get(values, *source)),
        TraceOp::Negate { source } => {
            Some(semantics::negate(get(values, *source)).expect("negate of a number"))
        }
        TraceOp::IntToFloat { source } => {
            Some(semantics::to_float(get(values, *source)).expect("conversion of an integer"))
        }
        TraceOp::Numeric { op, source } => Some(
            semantics::numeric(*op, get(values, *source)).expect("a numeric operation's operand"),
        ),
        TraceOp::ToWide {
            source,
            target,
            conversion,
        } => Some(
            semantics::to_wide(get(values, *source), *target, *conversion)
                .expect("conversion to a 64-bit integer"),
        ),
        TraceOp::NativeUnary { function, source } => {
            let TraceScalar::Float(argument) = get(values, *source) else {
                panic!("a native call takes a float");
            };
            Some(TraceScalar::Float((function.0)(argument)))
        }
        TraceOp::NativeBinary {
            function,
            left,
            right,
        } => {
            let (TraceScalar::Float(left), TraceScalar::Float(right)) =
                (get(values, *left), get(values, *right))
            else {
                panic!("a native call takes floats");
            };
            Some(TraceScalar::Float((function.0)(left, right)))
        }
        TraceOp::Arithmetic { op, left, right } => Some(
            semantics::arithmetic(*op, get(values, *left), get(values, *right))
                .expect("legal arithmetic"),
        ),
        TraceOp::Compare {
            comparison,
            left,
            right,
        } => Some(TraceScalar::Boolean(
            semantics::compare(*comparison, get(values, *left), get(values, *right))
                .expect("legal comparison"),
        )),
        TraceOp::Select {
            condition,
            if_true,
            if_false,
        } => {
            let TraceScalar::Boolean(condition) = get(values, *condition) else {
                panic!("a select on a non-boolean");
            };
            Some(get(values, if condition { *if_true } else { *if_false }))
        }
        TraceOp::Truthy { source } => Some(TraceScalar::Boolean(get(values, *source).is_truthy())),
        TraceOp::Not { source } => Some(TraceScalar::Boolean(!get(values, *source).is_truthy())),
        TraceOp::GuardType {
            source,
            expected,
            guard,
        } => {
            let scalar = get(values, *source);
            if scalar.ty() == *expected && !forced_at(*guard) {
                Some(scalar)
            } else {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            }
        }
        TraceOp::GuardBranch {
            condition,
            expected,
            guard,
        } => {
            let TraceScalar::Boolean(actual) = get(values, *condition) else {
                panic!("branch guard on a non-boolean");
            };
            if actual != *expected || forced_at(*guard) {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            }
            None
        }
        TraceOp::ArrayLoad {
            binding,
            key,
            expected,
            past_end,
            guard,
            ..
        } => {
            let loaded = if *expected == TraceType::Nil {
                (heap.array_hole(*binding, get(values, *key))
                    || (*past_end && heap.array_miss(*binding, get(values, *key))))
                .then_some(TraceScalar::Nil)
            } else {
                heap.array_get(*binding, get(values, *key))
                    .filter(|scalar| scalar.ty() == *expected)
            };
            match loaded {
                Some(scalar) if !forced_at(*guard) => Some(scalar),
                _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
            }
        }
        TraceOp::LaneArrayLoad {
            table,
            key,
            expected,
            guard,
            ..
        } => {
            let TraceScalar::Table(lane) = get(values, *table) else {
                panic!("a lane array read of a non-table");
            };
            match heap
                .lane_array_get(lane, get(values, *key))
                .filter(|scalar| scalar.ty() == *expected)
            {
                Some(scalar) if !forced_at(*guard) => Some(scalar),
                _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
            }
        }
        TraceOp::NumberLoad {
            binding,
            key,
            guard,
            ..
        } => {
            let loaded = match heap.array_get(*binding, get(values, *key)) {
                Some(TraceScalar::Integer(value)) if value.abs() <= 1 << 53 =>
                {
                    #[allow(clippy::cast_precision_loss)]
                    Some((TraceScalar::Float(value as f64), true))
                }
                Some(TraceScalar::Float(value)) => Some((TraceScalar::Float(value), false)),
                _ => None,
            };
            match loaded {
                Some((scalar, integer)) if !forced_at(*guard) => {
                    if let Some(result) = instruction.result {
                        NUMBER_TAGS.with(|tags| tags.borrow_mut().insert(result, integer));
                    }
                    Some(scalar)
                }
                _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
            }
        }
        TraceOp::NumberTag { load } => {
            Some(TraceScalar::Boolean(NUMBER_TAGS.with(|tags| {
                tags.borrow().get(load).copied().unwrap_or(false)
            })))
        }
        TraceOp::NumberModulo {
            float,
            left,
            right,
            integer,
        } => {
            let (
                TraceScalar::Float(left),
                TraceScalar::Float(right),
                TraceScalar::Boolean(integer),
            ) = (
                get(values, *left),
                get(values, *right),
                get(values, *integer),
            )
            else {
                panic!("a number modulo of non-floats");
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let result = if integer {
                super::semantics::integer_floor_modulo(left as i64, right as i64) as f64
            } else {
                match float {
                    super::ir::ArithmeticOp::ModuloWith(modulo) => modulo.apply(left, right),
                    _ => super::semantics::float_modulo(left, right),
                }
            };
            Some(TraceScalar::Float(result))
        }
        TraceOp::BoolAnd { left, right } => {
            let (TraceScalar::Boolean(left), TraceScalar::Boolean(right)) =
                (get(values, *left), get(values, *right))
            else {
                panic!("a conjunction of non-booleans");
            };
            Some(TraceScalar::Boolean(left && right))
        }
        TraceOp::ArrayNext {
            binding,
            key,
            start,
            ty,
            guard,
            ..
        } => match heap.array_next(*binding, get(values, *key), *start) {
            Some(index) if !forced_at(*guard) => Some(if *ty == TraceType::Float {
                #[allow(clippy::cast_precision_loss)]
                TraceScalar::Float(index as f64)
            } else {
                TraceScalar::Integer(index)
            }),
            _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
        },
        TraceOp::HashLoad {
            binding,
            key,
            expected,
            guard,
            ..
        } => match heap
            .hash_get(*binding, get(values, *key))
            .filter(|scalar| scalar.ty() == *expected)
        {
            Some(scalar) if !forced_at(*guard) => Some(scalar),
            _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
        },
        TraceOp::ReleaseStrings { .. } => None,
        TraceOp::HostCall {
            function,
            args,
            result,
            guard,
        } => {
            let args: Vec<TraceScalar> = args.iter().map(|arg| get(values, *arg)).collect();
            match heap
                .host_call(*function, &args)
                .filter(|scalar| scalar.ty() == *result)
            {
                Some(scalar) if !forced_at(*guard) => Some(scalar),
                _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
            }
        }
        TraceOp::CallHelper {
            helper,
            args,
            effect,
            guard,
        } => {
            let scalars: Vec<TraceScalar> = args.iter().map(|arg| get(values, *arg)).collect();
            let info = helper
                .info()
                .expect("a verified helper call names a helper");
            let answer = heap
                .helper_call(*helper, &scalars)
                .filter(|results| {
                    results.len() == info.results.len()
                        && results
                            .iter()
                            .zip(info.results)
                            .all(|(scalar, wanted)| scalar.ty() == wanted.trace_type())
                })
                .filter(|_| !forced_at(*guard));
            let Some(results) = answer else {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            };
            if let Some(effect) = effect {
                effects.push(EffectRecord {
                    effect: effect.0,
                    label: HELPER_EFFECT_LABEL_BASE + u32::from(helper.0),
                    operands: scalars,
                });
            }
            // The results after the first belong to the `HelperResult`
            // instructions that read them.
            if let Some(call) = instruction.result {
                for later in ir.preheader.iter().chain(&ir.body) {
                    if let (TraceOp::HelperResult { call: from, index }, Some(slot)) =
                        (&later.op, later.result)
                        && *from == call
                    {
                        values[slot.index()] = Some(results[usize::from(*index)]);
                    }
                }
            }
            results.first().copied()
        }
        // Read by the call that made it.
        TraceOp::HelperResult { .. } => instruction.result.and_then(|slot| values[slot.index()]),
        TraceOp::ArrayStore {
            binding,
            key,
            value,
            effect,
            guard,
            ..
        } => {
            let (key, value) = (get(values, *key), get(values, *value));
            if forced_at(*guard) || !heap.array_set(*binding, key, value) {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            }
            effects.push(EffectRecord {
                effect: effect.0,
                label: TABLE_STORE_LABEL,
                operands: vec![key, value],
            });
            None
        }
        // The verifier proved the slot holds a value of the stored type.
        TraceOp::ArrayOverwrite {
            binding,
            key,
            value,
            effect,
            ..
        } => {
            let (key, value) = (get(values, *key), get(values, *value));
            assert!(
                heap.array_set(*binding, key, value),
                "a proven array overwrite missed its slot"
            );
            effects.push(EffectRecord {
                effect: effect.0,
                label: TABLE_STORE_LABEL,
                operands: vec![key, value],
            });
            None
        }
        TraceOp::ArraySet {
            binding,
            key,
            value,
            effect,
            guard,
            ..
        } => {
            let (key, value) = (get(values, *key), get(values, *value));
            if forced_at(*guard) || !heap.table_set(*binding, key, value) {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            }
            effects.push(EffectRecord {
                effect: effect.0,
                label: TABLE_STORE_LABEL,
                operands: vec![key, value],
            });
            None
        }
        TraceOp::FieldLoad { binding, .. } => Some(heap.field_get(*binding)),
        TraceOp::FieldStore {
            binding,
            value,
            effect,
            ..
        } => {
            let value = get(values, *value);
            heap.field_set(*binding, value);
            effects.push(EffectRecord {
                effect: effect.0,
                label: TABLE_STORE_LABEL,
                operands: vec![value],
            });
            None
        }
        TraceOp::TableLength { binding, .. } => Some(heap.length(*binding)),
        TraceOp::StringLength { binding, .. } => Some(heap.string_length(*binding)),
        TraceOp::StringByte {
            binding,
            key,
            guard,
            ..
        } => match heap.string_byte(*binding, get(values, *key)) {
            Some(scalar) if !forced_at(*guard) => Some(scalar),
            _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
        },
        TraceOp::TableFieldLoad {
            field,
            table,
            guard,
        } => {
            let TraceScalar::Table(lane) = get(values, *table) else {
                panic!("a field read of a non-table");
            };
            let wanted = ir.fields[*field as usize].ty;
            match heap
                .table_field_get(lane, *field)
                .filter(|scalar| scalar.ty() == wanted)
            {
                Some(scalar) if !forced_at(*guard) => Some(scalar),
                _ => return Some((*guard, restore_guard(ir, *guard, values, registers, roots))),
            }
        }
        TraceOp::MetaGuard {
            table,
            binding,
            guard,
        } => {
            let TraceScalar::Table(lane) = get(values, *table) else {
                panic!("a metatable guard of a non-table");
            };
            let BindingKind::MetaClass { metatable, .. } = &ir.bindings[*binding as usize].kind
            else {
                panic!("a metatable guard names a metatable class");
            };
            let field = ir
                .fields
                .iter()
                .position(|field| field.key.as_ref() == META_FIELD_KEY)
                .expect("a verified metatable guard has the metatable field");
            let field = u32::try_from(field).expect("field index fits u32");
            // The row's word is the metatable's word, zero for none.
            let same = heap.table_field_get(lane, field)
                == Some(TraceScalar::Integer(
                    super::meta_word(*metatable).cast_signed(),
                ));
            if forced_at(*guard) || !same {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            }
            None
        }
        TraceOp::LaneArrayStore {
            table,
            key,
            value,
            effect,
            guard,
            ..
        } => {
            let TraceScalar::Table(lane) = get(values, *table) else {
                panic!("a lane array write of a non-table");
            };
            let value = get(values, *value);
            if forced_at(*guard) || !heap.lane_array_set(lane, get(values, *key), value) {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            }
            effects.push(EffectRecord {
                effect: effect.0,
                label: TABLE_STORE_LABEL,
                operands: vec![value],
            });
            None
        }
        TraceOp::TableFieldStore {
            field,
            table,
            value,
            effect,
            guard,
        } => {
            let TraceScalar::Table(lane) = get(values, *table) else {
                panic!("a field write of a non-table");
            };
            let value = get(values, *value);
            if forced_at(*guard) || !heap.table_field_set(lane, *field, value) {
                return Some((*guard, restore_guard(ir, *guard, values, registers, roots)));
            }
            effects.push(EffectRecord {
                effect: effect.0,
                label: TABLE_STORE_LABEL,
                operands: vec![value],
            });
            None
        }
        TraceOp::Read {
            label,
            operands,
            result,
        } => {
            let record = EffectRecord {
                effect: u32::MAX,
                label: *label,
                operands: operands
                    .iter()
                    .map(|operand| get(values, *operand))
                    .collect(),
            };
            let scalar = effect_result(&record, u32::try_from(effects.len()).unwrap())
                .expect("read result supplied");
            assert!(*result == TraceType::Any || scalar.ty() == *result);
            Some(scalar)
        }
        TraceOp::Effect {
            effect,
            label,
            operands,
            result,
        } => {
            let record = EffectRecord {
                effect: effect.0,
                label: *label,
                operands: operands
                    .iter()
                    .map(|operand| get(values, *operand))
                    .collect(),
            };
            let produced = result.map(|ty| {
                let scalar = effect_result(&record, u32::try_from(effects.len()).unwrap())
                    .expect("effect result supplied");
                // `Any` accepts every scalar; a concrete type must match.
                assert!(ty == TraceType::Any || scalar.ty() == ty);
                scalar
            });
            effects.push(record);
            produced
        }
    };
    if let (Some(result), Some(scalar)) = (instruction.result, produced) {
        values[result.index()] = Some(scalar);
    }
    None
}

fn restore_guard(
    ir: &TraceIr,
    guard: GuardId,
    values: &[Option<TraceScalar>],
    registers: &[TraceScalar],
    roots: &[TraceScalar],
) -> Restored {
    let snapshot = &ir.snapshots[ir.guards[guard.index()].snapshot.index()];
    restore(snapshot, values, registers, roots)
}

/// Builds the restored root-frame state, reading the innermost frame.
fn restore(
    snapshot: &Snapshot,
    values: &[Option<TraceScalar>],
    registers: &[TraceScalar],
    roots: &[TraceScalar],
) -> Restored {
    let integer_tag = |register: usize| -> bool {
        match snapshot.number_tag(register) {
            Some(RegisterEntry::Value(tag)) => get(values, tag) == TraceScalar::Boolean(true),
            Some(RegisterEntry::Const(scalar)) => scalar == TraceScalar::Boolean(true),
            _ => false,
        }
    };
    let registers_of = |frame: &super::snapshot::SnapshotFrame| -> Vec<TraceScalar> {
        frame
            .registers
            .iter()
            .enumerate()
            .map(|(index, entry)| match entry {
                RegisterEntry::Value(_) | RegisterEntry::Const(TraceScalar::Float(_))
                    if frame.frame == snapshot.frames[0].frame
                        && snapshot.number_tag(index).is_some()
                        && integer_tag(index) =>
                {
                    let TraceScalar::Float(float) = (match entry {
                        RegisterEntry::Value(value) => get(values, *value),
                        RegisterEntry::Const(scalar) => *scalar,
                        _ => unreachable!("matched above"),
                    }) else {
                        panic!("a number register holds a float");
                    };
                    #[allow(clippy::cast_possible_truncation)]
                    TraceScalar::Integer(float as i64)
                }
                RegisterEntry::Value(value) => get(values, *value),
                RegisterEntry::Const(scalar) => *scalar,
                RegisterEntry::Unchanged => registers[index],
                RegisterEntry::Root(slot) => roots[*slot as usize],
                RegisterEntry::Virtual(id) => {
                    TraceScalar::Table(VIRTUAL_LANE_BASE + u64::from(id.0))
                }
                RegisterEntry::Slice(id) => TraceScalar::String(SLICE_LANE_BASE + u64::from(id.0)),
            })
            .collect()
    };
    let resolve = |entry: &RegisterEntry| match entry {
        RegisterEntry::Value(value) => get(values, *value),
        RegisterEntry::Const(scalar) => *scalar,
        RegisterEntry::Root(slot) => roots[*slot as usize],
        // A table inside a table is stored as the lane that names it.
        RegisterEntry::Virtual(id) => TraceScalar::Table(VIRTUAL_LANE_BASE + u64::from(id.0)),
        RegisterEntry::Unchanged | RegisterEntry::Slice(_) => {
            panic!("a verified recipe holds values, constants, roots and tables")
        }
    };
    let mut named: Vec<VirtualId> = snapshot
        .frames
        .iter()
        .flat_map(|frame| &frame.registers)
        .filter_map(|entry| match entry {
            RegisterEntry::Virtual(id) => Some(*id),
            _ => None,
        })
        .collect();
    // The tables those hold are built too.
    let mut position = 0;
    while position < named.len() {
        let inner: Vec<VirtualId> = snapshot.virtuals[named[position].index()]
            .entries()
            .filter_map(|entry| match entry {
                RegisterEntry::Virtual(id) => Some(*id),
                _ => None,
            })
            .collect();
        for id in inner {
            if !named.contains(&id) {
                named.push(id);
            }
        }
        position += 1;
    }
    named.sort_by_key(|id| id.0);
    named.dedup();
    let tables = named
        .into_iter()
        .map(|id| {
            let recipe = &snapshot.virtuals[id.index()];
            (
                id,
                RestoredTable {
                    alloc: recipe.alloc.clone(),
                    array: recipe.array.iter().map(resolve).collect(),
                    fields: recipe
                        .fields
                        .iter()
                        .map(|(key, entry)| (key.clone(), resolve(entry)))
                        .collect(),
                    metatable: recipe.metatable.as_ref().map(resolve),
                },
            )
        })
        .collect();
    let (innermost, outer) = snapshot.frames.split_last().expect("snapshot has a frame");
    let resume = snapshot.resume_point();
    Restored {
        pc: resume.pc,
        last_interpreted_pc: resume.last_interpreted_pc,
        registers: registers_of(innermost),
        effect_position: snapshot.effect_position,
        tables,
        outer: outer
            .iter()
            .map(|frame| RestoredFrame {
                frame: frame.frame,
                last_interpreted_pc: frame.last_interpreted_pc,
                registers: registers_of(frame),
            })
            .collect(),
    }
}

/// What [`evaluate_tree`] did.
#[derive(Debug)]
pub struct TreeEvaluation {
    /// How the last trace of the chain ended.
    pub outcome: Outcome,
    /// Loop-backs completed in the root and links taken from sides.
    pub iterations: u32,
    /// Side traces entered, counting each entry.
    pub side_entries: u32,
}

/// Names one guard of a tree: the side it belongs to (`None` for the root)
/// as an index into [`TraceIr::sides`], and its [`super::GuardInfo::origin`]
/// in that trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TreeGuard {
    pub side: Option<usize>,
    pub origin: u32,
}

/// Evaluates a root trace and its side traces as the interpreter would run
/// them: when a guard that has a side fails, the state it restores is the
/// state the side starts from, and when a side links back, the state it
/// publishes is the state the root re-enters at its loop header. Nothing here
/// knows about widening, optimization or native links, so it is the oracle a
/// compiled tree must agree with.
///
/// `max_iterations` bounds loop-backs and links together.
#[must_use]
pub fn evaluate_tree(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    max_iterations: u32,
) -> TreeEvaluation {
    evaluate_tree_forced(ir, registers, roots, max_iterations, None)
}

/// [`evaluate_tree`] with one guard forced to fail whenever it runs, the way
/// a compiled tree is built to take that guard's exit (or its side) at once.
/// A guard that has a side continues in it, and one that has none ends the
/// evaluation.
pub fn evaluate_tree_forced(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    max_iterations: u32,
    forced: Option<TreeGuard>,
) -> TreeEvaluation {
    evaluate_tree_with_heap(ir, registers, roots, max_iterations, forced, &mut NoHeap)
}

/// [`evaluate_tree_forced`] for a tree with table operations, over a model of
/// the tables its bindings resolve. The model is indexed as the tree's
/// shared binding list is (see [`TraceIr::bindings`]): a side lists the
/// root's bindings first, so one model serves every member, as long as it
/// covers the longest list.
pub fn evaluate_tree_with_heap(
    ir: &TraceIr,
    registers: &[TraceScalar],
    roots: &[TraceScalar],
    max_iterations: u32,
    forced: Option<TreeGuard>,
    heap: &mut dyn TraceHeap,
) -> TreeEvaluation {
    let mut root = ir.clone();
    root.sides.clear();
    let mut state = registers.to_vec();
    let mut iterations = 0_u32;
    let mut side_entries = 0_u32;
    let no_effects = |_: &EffectRecord, _: u32| None;
    // The forced exit of the trace `member` (`None` is the root), if the
    // forced guard is one of its guards.
    let forcing = |member: Option<usize>, trace: &TraceIr| -> Option<ForcedExit> {
        let forced = forced.filter(|forced| forced.side == member)?;
        let guard = trace
            .guards
            .iter()
            .position(|info| info.origin == forced.origin)?;
        Some(ForcedExit {
            frame: FrameId(0),
            pc: trace.guards[guard].site.pc,
            guard: Some(GuardId(u32::try_from(guard).expect("guard fits u32"))),
            iteration: 0,
        })
    };
    // The trace whose guard just failed: `None` is the root.
    'enter: loop {
        let left = max_iterations.saturating_sub(iterations).max(1);
        let evaluation = evaluate_with_heap(
            &root,
            &state,
            roots,
            left,
            forcing(None, &root),
            heap,
            no_effects,
        );
        iterations += evaluation.iterations;
        let mut current: Option<usize> = None;
        let mut outcome = evaluation.outcome;
        loop {
            // A side (or a root) that ended in front of a loop hands it the
            // state its terminal exit restored.
            if let Outcome::Exited(restored) = &outcome
                && matches!(
                    current.map_or(&root.terminator, |from| &ir.sides[from].ir.terminator),
                    TraceTerminator::Exit { .. }
                )
                && let Some(side) = ir
                    .sides
                    .iter()
                    .position(|side| side.parent == current && side.guard == TERMINAL_GUARD)
            {
                side_entries += 1;
                let known = restored.registers.len();
                state.splice(..known.min(state.len()), restored.registers.iter().copied());
                if state.len() < known {
                    state = restored.registers.clone();
                }
                let member = &ir.sides[side].ir;
                let evaluation = evaluate_with_heap(
                    member,
                    &state,
                    roots,
                    left,
                    forcing(Some(side), member),
                    heap,
                    no_effects,
                );
                iterations += evaluation.iterations;
                current = Some(side);
                outcome = evaluation.outcome;
                continue;
            }
            let Outcome::GuardFailed { guard, restored } = &outcome else {
                return TreeEvaluation {
                    outcome,
                    iterations,
                    side_entries,
                };
            };
            let source = match current {
                None => &root,
                Some(side) => &ir.sides[side].ir,
            };
            let origin = source.guards[guard.index()].origin;
            let Some(side) = ir
                .sides
                .iter()
                .position(|side| side.parent == current && side.guard == origin)
            else {
                return TreeEvaluation {
                    outcome,
                    iterations,
                    side_entries,
                };
            };
            side_entries += 1;
            // A window slot the failing trace does not name (a cell or global
            // only a side does) keeps the value it had.
            let known = restored.registers.len();
            state.splice(..known.min(state.len()), restored.registers.iter().copied());
            if state.len() < known {
                state = restored.registers.clone();
            }
            let member = &ir.sides[side].ir;
            let evaluation = evaluate_with_heap(
                member,
                &state,
                roots,
                left,
                forcing(Some(side), member),
                heap,
                no_effects,
            );
            current = Some(side);
            outcome = evaluation.outcome;
            // A side that rejoins the loop of a loop side it descends from
            // runs that loop on: one more loop-back of it, from the state
            // the link names.
            while let Outcome::Exited(restored) = &outcome
                && let Some(from) = current
                && let TraceTerminator::LinkLoop { side: target, .. } = ir.sides[from].ir.terminator
            {
                iterations += 1;
                if iterations >= max_iterations {
                    return TreeEvaluation {
                        outcome: Outcome::IterationLimit { iterations },
                        iterations,
                        side_entries,
                    };
                }
                let known = restored.registers.len();
                state.splice(..known.min(state.len()), restored.registers.iter().copied());
                if state.len() < known {
                    state = restored.registers.clone();
                }
                let member = &ir.sides[target].ir;
                let evaluation = evaluate_with_heap(
                    member,
                    &state,
                    roots,
                    left,
                    forcing(Some(target), member),
                    heap,
                    no_effects,
                );
                iterations += evaluation.iterations;
                current = Some(target);
                outcome = evaluation.outcome;
            }
            if let Outcome::Exited(restored) = &outcome
                && let Some(side) = current
                && matches!(ir.sides[side].ir.terminator, TraceTerminator::Link { .. })
            {
                // The side rejoined the loop: that is one more loop-back.
                iterations += 1;
                if iterations >= max_iterations {
                    return TreeEvaluation {
                        outcome: Outcome::IterationLimit { iterations },
                        iterations,
                        side_entries,
                    };
                }
                let known = restored.registers.len();
                state.splice(..known.min(state.len()), restored.registers.iter().copied());
                if state.len() < known {
                    state = restored.registers.clone();
                }
                continue 'enter;
            }
        }
    }
}
