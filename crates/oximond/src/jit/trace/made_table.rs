//! The tables a [`HelperId::TABLE_NEW`] call makes and the calls of
//! [`HelperId::TABLE_INIT`] that fill them, as both native backends decide
//! them.
//!
//! The helper leaves the address of the new table's first value slot in its
//! second result word when the code is going to fill the table itself, and
//! the code then stores each field where it is. These functions answer which
//! tables that applies to and which slot a call fills. They read only the
//! trace and the constants lowered so far, so the two backends take the same
//! tables and give the same exit behavior.

use crate::jit_ir::trace::{
    HelperId, MAX_PREFILLED_ARRAY, ROOT_STRING_LANE_BASE, TraceIr, TraceOp, TraceScalar, TraceType,
    ValueId,
};
use crate::runtime::TableShape;
use std::sync::Arc;

/// What a [`HelperId::TABLE_NEW`] call told the code about the table it made.
#[derive(Clone)]
pub(super) struct MadeShape {
    /// The shape the call named, which gives each key its slot.
    pub(super) shape: Option<Arc<TableShape>>,
    /// Whether the call made the array part `array` nil slots long, which
    /// the code fills in place.
    pub(super) prefilled: bool,
    /// The array slots the call made room for, when the length is a
    /// constant. A store inside them moves nothing.
    pub(super) array: i64,
}

/// Whether some [`HelperId::TABLE_INIT`] of `ir` names the table `made` by a
/// root string key, which the code can fill in place.
pub(super) fn fills_in_place(ir: &TraceIr, made: ValueId) -> bool {
    ir.body.iter().any(|instruction| {
        matches!(
            &instruction.op,
            TraceOp::CallHelper { helper: HelperId::TABLE_INIT, args, .. }
                if args.first() == Some(&made)
                    && args.get(1).is_some_and(|key| {
                        ir.values[key.index()].ty == TraceType::String
                    })
        )
    })
}

/// Whether the code fills the table `made`, which a [`HelperId::TABLE_NEW`]
/// call with arguments `args` makes, as a short array in place: the call
/// names a constant length of [`MAX_PREFILLED_ARRAY`] or fewer and no key,
/// and the trace stores a number or boolean under each integer key from one
/// to the length, once, by calls of [`HelperId::TABLE_INIT`]. The helper then
/// makes the slots (nil, counted as present), and each store overwrites one,
/// so the table is the dense list the constructor makes.
pub(super) fn prefills_array(
    ir: &TraceIr,
    made: ValueId,
    args: &[ValueId],
    constant_integer: impl Fn(&TraceIr, ValueId) -> Option<i64>,
) -> bool {
    let Some(length) = args
        .first()
        .and_then(|length| constant_integer(ir, *length))
        .and_then(|length| usize::try_from(length).ok())
        .filter(|length| (1..=MAX_PREFILLED_ARRAY).contains(length))
    else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    let mut stored = vec![false; length];
    for instruction in &ir.body {
        let TraceOp::CallHelper {
            helper: HelperId::TABLE_INIT,
            args: init,
            ..
        } = &instruction.op
        else {
            continue;
        };
        if init.first() != Some(&made) {
            continue;
        }
        let Some(key) = constant_integer(ir, init[1])
            .and_then(|key| usize::try_from(key).ok())
            .filter(|key| (1..=length).contains(key))
        else {
            return false;
        };
        if stored[key - 1]
            || !matches!(
                ir.value_type(init[2]),
                Some(TraceType::Float | TraceType::Integer | TraceType::Boolean)
            )
        {
            return false;
        }
        stored[key - 1] = true;
    }
    stored.iter().all(|stored| *stored)
}

/// The shape a [`HelperId::TABLE_NEW`] call with arguments `args` names, when
/// every key is a root string of the trace.
pub(super) fn made_table_shape(
    ir: &TraceIr,
    args: &[ValueId],
    constants: &[Option<TraceScalar>],
) -> Option<Arc<TableShape>> {
    let mut keys: Vec<&[u8]> = Vec::new();
    for arg in args.iter().skip(1) {
        let root = root_key(ir, constants[arg.index()])?;
        keys.push(root);
    }
    TableShape::from_bytes(keys)
}

/// The root string a constant key names.
fn root_key(ir: &TraceIr, key: Option<TraceScalar>) -> Option<&[u8]> {
    let Some(TraceScalar::String(lane)) = key else {
        return None;
    };
    let root = usize::try_from(lane.checked_sub(ROOT_STRING_LANE_BASE)?).ok()?;
    ir.root_constants.get(root).map(|key| &**key)
}

/// The slot a [`HelperId::TABLE_INIT`] of the key `key` fills in the table
/// `made` describes, if the code can fill it in place: the array slot of an
/// integer key the call made room for, or the slot the shape names for a root
/// string key.
pub(super) fn init_field(
    ir: &TraceIr,
    made: &MadeShape,
    key: Option<TraceScalar>,
) -> Option<usize> {
    if made.prefilled {
        let Some(TraceScalar::Integer(key)) = key else {
            return None;
        };
        return match usize::try_from(key) {
            Ok(key) if (1..=usize::try_from(made.array).unwrap_or(0)).contains(&key) => {
                Some(key - 1)
            }
            _ => None,
        };
    }
    let shape = made.shape.as_ref()?;
    shape.slot_of(root_key(ir, key)?)
}

/// Whether a [`HelperId::TABLE_INIT`] that the helper answers may move the
/// slots of the table `made` describes: a key the shape lacks (an array
/// element, say) can grow the table's storage, and a nil value deletes the
/// slot's entry instead of filling it. Later in-place stores must not trust
/// the slots' address then.
pub(super) fn init_by_helper_moves(
    ir: &TraceIr,
    made: &MadeShape,
    key: Option<TraceScalar>,
    value: ValueId,
) -> bool {
    let in_array = matches!(
        key,
        Some(TraceScalar::Integer(index)) if (1..=made.array).contains(&index)
    );
    let in_shape = in_array
        || made
            .shape
            .as_ref()
            .is_some_and(|shape| root_key(ir, key).is_some_and(|key| shape.slot_of(key).is_some()));
    let never_nil = ir.value_type(value).is_some_and(|ty| {
        matches!(
            ty,
            TraceType::Float
                | TraceType::Integer
                | TraceType::Boolean
                | TraceType::Table
                | TraceType::String
        )
    });
    !in_shape || !never_nil
}
