//! Restores a trace snapshot onto a real VM frame.
//!
//! A failed guard, a completed loop or a poll leaves native code with typed
//! SSA values and a snapshot id. [`restore_root_snapshot`] turns those into
//! the exact interpreter state the snapshot describes: register values,
//! `pc`, and `last_interpreted_pc`. It uses the frame the interpreter
//! already owns and never synthesizes a new one.
//!
//! The phase rules:
//!
//! - `Before { pc }` resumes at `pc`. `last_interpreted_pc` keeps the
//!   previously executed pc the snapshot recorded.
//! - `After { pc, successor }` resumes at `successor`. The instruction at
//!   `pc` committed exactly once, so `last_interpreted_pc` is `pc`. Tracing
//!   backedge detection reads that field on the next dispatch.
//!
//! Typed values keep their type: integer bits stay `Value::Integer` and float
//! bits stay `Value::Number`. Nothing goes through `f64` unless the value is
//! a float.
//!
//! # Captured locals, upvalues and opaque values
//!
//! A captured local of the frame lives in its register, and its open cell
//! reads and writes that register. The restore here writes registers, which
//! restores those cells too; nothing has to be reconciled around a run.
//!
//! The window a snapshot describes may be longer than the frame's registers:
//! the trace's upvalue and global slots follow them (see
//! [`oximond::jit_ir::trace::ExternalSlot`]). A restore returns the cell writes
//! its upvalue slots call for instead of performing them, since the frame
//! has no access to the runtime. A global slot is never written.
//!
//! Tables and closures arrive as lane ids and resolve through the run's root
//! bank. `Root` entries resolve through the trace's constant roots.
//!
//! # Virtual tables
//!
//! A `Virtual` entry names a table the trace never allocated.
//! [`materialize_virtuals`] builds the ones a snapshot names, and the ones
//! those hold (a recipe may store another recipe's table), all allocated
//! before any is filled, and the restore then writes the handle into every
//! register that names the id. Nothing in between reaches a safepoint, so the
//! collector never sees a partial table. A recipe whose table the trace
//! already charged as phantom debt gives the charge back, so the table counts
//! once (see `Runtime::refund_elided_table`). The native plan in `native.rs`
//! does the same from its prepared `VirtualPlan`s.
//!
//! # Inlined calls
//!
//! A snapshot taken inside an inlined call lists the root frame and each
//! callee, outermost first. [`restore_snapshot`] rebuilds them in call
//! order: the root frame the interpreter already owns stops at its call (its
//! `pc` is the instruction after, its `last_interpreted_pc` the call), and
//! each callee is pushed with the activation record the interpreter's own
//! call would have made, from the pinned closure's chunk, cells and
//! environment, returning to the call's result window. The innermost frame
//! resumes at the snapshot's phase. The interpreter then continues the call
//! exactly once, and a traceback or `debug.getinfo` from the callee sees an
//! ordinary frame chain.
//!
//! Restoration is all or nothing. It checks every entry before it writes, so
//! an error leaves the frame as the interpreter had it.
//!
//! # Tail calls of the root frame
//!
//! A trace whose loop runs through tail calls of the root frame (see
//! [`oximond::jit_ir::trace::TraceIr::loop_tail`]) may leave while another
//! closure runs in the root frame: a later activation
//! ([`oximond::jit_ir::trace::RootActivation`]). [`apply_root_tail`] does to the
//! frame what the interpreter's in-place tail calls would have done: it
//! replaces the frame's closure, chunk, cells and environment, clears its
//! interior state, resizes its registers to the chunk's, and takes the tail
//! call count of the [`ExternalSlot::TailCalls`] slot and the name of the last
//! tail call. A count the run did not change (an exit before the first tail
//! call) leaves the frame's name and count as the entry found them.

use crate::jit_ir::trace::{
    Entry, ExternalSlot, FrameId, Protected, ROOT_STRING_LANE_BASE, RegisterEntry, ResumePoint,
    Snapshot, TailSite, TraceIr, TraceScalar, ValueId, VirtualAlloc, VirtualId, VirtualKey,
    VirtualTable,
};
use crate::runtime::CaptureCells;
use crate::runtime::{ClosureHandle, Runtime, RuntimeError, TableHandle, TableShape, Value};
use crate::string::LuaString;
use crate::vm::ConcatRangeContinuation;
use crate::vm::frame::FrameChunk;
use crate::vm::stack::{ActivationArguments, CallStack, FrameMut};
use crate::vm::{
    CallReturn, ExtendedProtectedCall, ReturnTarget, ScalarContinuation, ToStringCompletion,
    ToStringContinuation,
};
use std::fmt;

/// Where restoration reads SSA values from. Native exits implement this over
/// their typed exit area, using the exit descriptor to decode each slot.
pub(crate) trait ExitValues {
    /// The scalar a value held when the trace left, if it was spilled.
    fn scalar(&self, value: ValueId) -> Option<TraceScalar>;
}

impl ExitValues for [Option<TraceScalar>] {
    fn scalar(&self, value: ValueId) -> Option<TraceScalar> {
        self.get(value.index()).copied().flatten()
    }
}

/// Why a snapshot could not be restored onto a frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DeoptError {
    /// Only root-frame snapshots restore through this path.
    NotRootOnly { frames: usize },
    /// The snapshot's first frame is not the root frame.
    NotRootFrame(FrameId),
    /// The register map is not exactly the frame's register window.
    RegisterCount { snapshot: usize, frame: usize },
    /// A global slot was mapped to something other than `Unchanged`.
    GlobalWritten,
    /// A lane id names no value of the bank, or one of another type.
    MissingLane(u64),
    /// The resume or last-interpreted pc is outside the frame's chunk.
    PcOutOfBounds(usize),
    /// A register names a value the exit did not provide.
    MissingValue(ValueId),
    /// A register names a root slot the bank does not have.
    MissingRoot(u32),
    /// A snapshot frame is not an inlined call of the trace, or has a window
    /// of another size.
    BadInlinedFrame(FrameId),
    /// The pinned closure of an inlined frame is not in the bank or no
    /// longer exists.
    MissingClosure(u32),
    /// A register names a virtual table the caller did not materialize.
    MissingVirtual(VirtualId),
    /// A virtual table recipe cannot be built: the allocation failed, or the
    /// recipe is malformed.
    Materialize(String),
    /// The root frame's tail call count is not an integer.
    TailCount,
}

impl fmt::Display for DeoptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRootOnly { frames } => {
                write!(f, "snapshot has {frames} frames, root restore needs one")
            }
            Self::NotRootFrame(frame) => write!(f, "snapshot starts at frame {}", frame.0),
            Self::RegisterCount { snapshot, frame } => write!(
                f,
                "snapshot maps {snapshot} registers for a window of {frame}"
            ),
            Self::GlobalWritten => write!(f, "a global slot would be written"),
            Self::MissingLane(lane) => write!(f, "root bank has no matching lane {lane}"),
            Self::PcOutOfBounds(pc) => write!(f, "pc {pc} is outside the chunk"),
            Self::MissingValue(value) => write!(f, "exit has no value v{}", value.0),
            Self::MissingRoot(slot) => write!(f, "root bank has no slot {slot}"),
            Self::BadInlinedFrame(frame) => {
                write!(f, "frame {} is not an inlined call of the trace", frame.0)
            }
            Self::MissingClosure(lane) => write!(f, "pinned closure {lane} is not available"),
            Self::MissingVirtual(id) => write!(f, "virtual table {} was not materialized", id.0),
            Self::Materialize(reason) => write!(f, "cannot materialize a virtual table: {reason}"),
            Self::TailCount => write!(f, "the tail call count is not an integer"),
        }
    }
}

impl std::error::Error for DeoptError {}

/// The VM value a plain scalar stands for. A table or closure lane has none
/// without the run's root bank, and a verified trace has no such constant.
pub(crate) const fn scalar_to_value(scalar: TraceScalar) -> Value {
    match scalar {
        TraceScalar::Nil => Value::Nil,
        TraceScalar::Boolean(value) => Value::Boolean(value),
        TraceScalar::Integer(value) => Value::Integer(value),
        TraceScalar::Float(value) => Value::Number(value),
        TraceScalar::Int64(value) => Value::Int64(value),
        TraceScalar::UInt64(value) => Value::UInt64(value),
        TraceScalar::Table(_)
        | TraceScalar::Closure(_)
        | TraceScalar::Thread(_)
        | TraceScalar::Userdata(_)
        | TraceScalar::String(_) => {
            panic!("an opaque value resolves through the root bank")
        }
        TraceScalar::StringRef(_) => panic!("a borrowed string resolves through its slot"),
    }
}

/// The owned string value a borrowed string stands for: a clone of the string
/// held in the table slot whose address is `bits`.
///
/// # Safety
///
/// `bits` must be the address of a live, initialized [`Value`] that a trace
/// borrowed with [`oximond::jit_ir::trace::TraceOp::ArrayLoad`] expecting
/// [`oximond::jit_ir::trace::TraceType::StringRef`], in this run. The contract
/// that makes this hold: the borrow's owner is a table bound at entry, so it
/// is rooted for the whole run; the slot lies in the table's array part, which
/// nothing in a run resizes or replaces except operations for which
/// [`oximond::jit_ir::trace::TraceOp::ends_borrows`] holds; and the verifier
/// refuses a trace that uses a borrow after one of those. The restore runs
/// before any code that could collect or mutate a table.
#[allow(unsafe_code)]
pub(crate) unsafe fn clone_borrowed_string(bits: u64) -> Value {
    // SAFETY: the caller promises `bits` addresses a live `Value`.
    let slot = unsafe { &*(bits as usize as *const Value) };
    if let Value::String(text) = slot {
        Value::String(text.clone())
    } else {
        // The slot's tag was checked by the load that made the borrow.
        debug_assert!(false, "a borrowed slot holds a string");
        Value::Nil
    }
}

/// The typed scalar for a VM value, if the trace IR can represent it.
pub(crate) const fn value_to_scalar(value: &Value) -> Option<TraceScalar> {
    match value {
        Value::Nil => Some(TraceScalar::Nil),
        Value::Boolean(value) => Some(TraceScalar::Boolean(*value)),
        Value::Integer(value) => Some(TraceScalar::Integer(*value)),
        Value::Number(value) => Some(TraceScalar::Float(*value)),
        Value::Int64(value) => Some(TraceScalar::Int64(*value)),
        Value::UInt64(value) => Some(TraceScalar::UInt64(*value)),
        _ => None,
    }
}

/// Where one register's new value comes from.
enum Source {
    Keep,
    Value(Value),
}

/// The VM value a scalar stands for, resolving a table or closure lane
/// through `bank`.
fn resolve(scalar: TraceScalar, bank: &[Value], roots: &[Value]) -> Result<Value, DeoptError> {
    match scalar {
        TraceScalar::Table(lane)
        | TraceScalar::Closure(lane)
        | TraceScalar::Thread(lane)
        | TraceScalar::Userdata(lane)
        | TraceScalar::String(lane) => {
            // A string lane from the root base on names a constant of the
            // trace's roots.
            let held = match (scalar, lane.checked_sub(ROOT_STRING_LANE_BASE)) {
                (TraceScalar::String(_), Some(slot)) => {
                    usize::try_from(slot).ok().and_then(|slot| roots.get(slot))
                }
                _ => usize::try_from(lane).ok().and_then(|index| bank.get(index)),
            };
            held.filter(|value| match scalar {
                TraceScalar::Table(_) => matches!(value, Value::Table(_)),
                TraceScalar::String(_) => matches!(value, Value::String(_)),
                TraceScalar::Thread(_) => matches!(value, Value::Thread(_)),
                TraceScalar::Userdata(_) => matches!(value, Value::Userdata(_)),
                _ => matches!(
                    value,
                    Value::Closure(_)
                        | Value::CoroutineWrapper(_)
                        | Value::NativeFunction(_)
                        | Value::BuiltinFunction(_)
                        | Value::GmatchIterator(_)
                ),
            })
            .cloned()
            .ok_or(DeoptError::MissingLane(lane))
        }
        // SAFETY: a borrow in an exit area was made by this run's trace; see
        // `clone_borrowed_string`.
        #[allow(unsafe_code)]
        TraceScalar::StringRef(bits) => Ok(unsafe { clone_borrowed_string(bits) }),
        scalar => Ok(scalar_to_value(scalar)),
    }
}

/// Where a restore writes a cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CellTarget {
    /// Upvalue `index` of the root frame's closure.
    Frame(u16),
    /// Upvalue `index` of pinned closure `lane`.
    Closure { lane: u32, index: u16 },
    /// Register `index` of the frame below the root frame (the caller). It is
    /// no cell: the write goes into the caller's register window.
    Caller(u16),
}

/// What a restore leaves for the caller to do.
#[derive(Debug)]
pub(crate) struct RootRestore {
    pub(crate) resume: ResumePoint,
    /// The values to write into upvalue cells.
    pub(crate) upvalues: Vec<(CellTarget, Value)>,
}

/// The cell writes the window slots beyond a frame's registers call for, from
/// their resolved sources.
fn cell_writes(
    externals: &[ExternalSlot],
    resolved: impl Iterator<Item = Source>,
) -> Result<Vec<(CellTarget, Value)>, DeoptError> {
    let mut writes = Vec::new();
    for (slot, source) in externals.iter().zip(resolved) {
        match (slot, source) {
            // The count goes to the frame (see `apply_root_tail`).
            (_, Source::Keep) | (ExternalSlot::TailCalls, _) => {}
            (ExternalSlot::CallerRegister(register), Source::Value(value)) => {
                writes.push((CellTarget::Caller(*register), value));
            }
            (ExternalSlot::Upvalue(index), Source::Value(value)) => {
                writes.push((CellTarget::Frame(*index), value));
            }
            (ExternalSlot::ClosureUpvalue { closure, index }, Source::Value(value)) => {
                writes.push((
                    CellTarget::Closure {
                        lane: *closure,
                        index: *index,
                    },
                    value,
                ));
            }
            (
                ExternalSlot::Global(_)
                | ExternalSlot::ClosureGlobal { .. }
                | ExternalSlot::CallerGlobal(_)
                | ExternalSlot::UpvalueField { .. }
                | ExternalSlot::ClosureUpvalueField { .. }
                | ExternalSlot::RootClosure
                | ExternalSlot::VarargCount
                | ExternalSlot::Vararg(_)
                | ExternalSlot::Environment,
                Source::Value(_),
            ) => {
                return Err(DeoptError::GlobalWritten);
            }
        }
    }
    Ok(writes)
}

/// Resolves every entry of a register map against `values`, the lane bank
/// and the roots, without writing anything.
fn resolve_registers(
    entries: &crate::jit_ir::trace::RegisterMap,
    numbers: &[crate::jit_ir::trace::NumberRegister],
    values: &(impl ExitValues + ?Sized),
    bank: &[Value],
    roots: &[Value],
    virtuals: &[Option<TableHandle>],
) -> Result<Vec<Source>, DeoptError> {
    // The integer a register of run-time subtype holds, if it does.
    let is_integer = |register: usize| -> Result<bool, DeoptError> {
        let Some(number) = numbers
            .iter()
            .find(|number| number.register as usize == register)
        else {
            return Ok(false);
        };
        match number.is_integer {
            RegisterEntry::Const(TraceScalar::Boolean(integer)) => Ok(integer),
            RegisterEntry::Value(value) => match values.scalar(value) {
                Some(TraceScalar::Boolean(integer)) => Ok(integer),
                _ => Err(DeoptError::MissingValue(value)),
            },
            _ => Err(DeoptError::Materialize(
                "a number tag is a boolean value or constant".to_owned(),
            )),
        }
    };
    entries
        .iter()
        .enumerate()
        .map(|(register, entry)| {
            Ok(match entry {
                RegisterEntry::Const(_) | RegisterEntry::Value(_) if is_integer(register)? => {
                    let float = match entry {
                        RegisterEntry::Const(scalar) => *scalar,
                        RegisterEntry::Value(value) => values
                            .scalar(*value)
                            .ok_or(DeoptError::MissingValue(*value))?,
                        _ => unreachable!("matched above"),
                    };
                    let TraceScalar::Float(float) = float else {
                        return Err(DeoptError::Materialize(
                            "a number register holds a float".to_owned(),
                        ));
                    };
                    #[allow(clippy::cast_possible_truncation)]
                    Source::Value(Value::Integer(float as i64))
                }
                RegisterEntry::Unchanged => Source::Keep,
                RegisterEntry::Const(scalar) => Source::Value(resolve(*scalar, bank, roots)?),
                RegisterEntry::Value(value) => Source::Value(resolve(
                    values
                        .scalar(*value)
                        .ok_or(DeoptError::MissingValue(*value))?,
                    bank,
                    roots,
                )?),
                RegisterEntry::Root(slot) => Source::Value(
                    roots
                        .get(*slot as usize)
                        .ok_or(DeoptError::MissingRoot(*slot))?
                        .clone(),
                ),
                RegisterEntry::Virtual(id) => Source::Value(Value::Table(
                    virtuals
                        .get(id.index())
                        .copied()
                        .flatten()
                        .ok_or(DeoptError::MissingVirtual(*id))?,
                )),
                // Only the exit plan (`ExitRestore`) makes a substring.
                RegisterEntry::Slice(_) => {
                    return Err(DeoptError::Materialize(
                        "the generic restore does not make a substring".to_owned(),
                    ));
                }
            })
        })
        .collect()
}

/// The VM value a recipe entry stands for: a snapshot value, constant or
/// root. A table inside a table is [`resolve_item`]'s. A recipe never holds
/// `Unchanged`.
fn resolve_recipe_entry(
    entry: &RegisterEntry,
    values: &(impl ExitValues + ?Sized),
    bank: &[Value],
    roots: &[Value],
) -> Result<Value, DeoptError> {
    match entry {
        RegisterEntry::Const(scalar) => resolve(*scalar, bank, roots),
        RegisterEntry::Value(value) => resolve(
            values
                .scalar(*value)
                .ok_or(DeoptError::MissingValue(*value))?,
            bank,
            roots,
        ),
        RegisterEntry::Root(slot) => roots
            .get(*slot as usize)
            .cloned()
            .ok_or(DeoptError::MissingRoot(*slot)),
        RegisterEntry::Unchanged | RegisterEntry::Virtual(_) | RegisterEntry::Slice(_) => Err(
            DeoptError::Materialize("a recipe holds only values, constants and roots".to_owned()),
        ),
    }
}

/// The number key `index` is in the runtime's profile: an integer where the
/// profile has them, otherwise the float of the same value.
#[must_use]
pub(crate) fn integer_key(runtime: &Runtime, index: i64) -> Value {
    if runtime.tagged_integers() {
        Value::Integer(index)
    } else {
        #[allow(clippy::cast_precision_loss)] // A profile without integers has no wider key.
        Value::Number(index as f64)
    }
}

/// The key value of a recipe field.
#[must_use]
pub(crate) fn virtual_key_value(runtime: &Runtime, key: &VirtualKey) -> Value {
    match key {
        VirtualKey::Integer(index) => integer_key(runtime, *index),
        VirtualKey::Float(number) => Value::Number(*number),
        VirtualKey::String(bytes) => Value::String(LuaString::from(&bytes[..])),
    }
}

/// The shape a shaped recipe allocates with, or `None` for a plain one (or
/// for keys no shape can hold).
#[must_use]
pub(crate) fn virtual_shape(alloc: &VirtualAlloc) -> Option<std::sync::Arc<TableShape>> {
    match alloc {
        VirtualAlloc::Plain => None,
        VirtualAlloc::Shaped { keys } => {
            TableShape::new(keys.iter().map(|key| LuaString::from(&key[..])))
        }
    }
}

/// Allocates one table for a recipe's allocation kind.
pub(crate) fn allocate_virtual(
    runtime: &mut Runtime,
    shape: Option<&std::sync::Arc<TableShape>>,
) -> Result<TableHandle, RuntimeError> {
    match shape {
        Some(shape) => runtime.create_table_with_shape(shape),
        None => runtime.create_table(),
    }
}

/// [`allocate_virtual`] for a recipe with `array` values at keys 1, 2, ...:
/// a plain table has room for them, as the table the interpreter's
/// constructor makes has. The room is capacity only, so the table is the
/// one [`allocate_virtual`] makes once it is filled.
pub(crate) fn allocate_virtual_sized(
    runtime: &mut Runtime,
    shape: Option<&std::sync::Arc<TableShape>>,
    array: usize,
) -> Result<TableHandle, RuntimeError> {
    match shape {
        Some(shape) => runtime.create_table_with_shape(shape),
        None => runtime.create_table_sized(array.min(64)),
    }
}

/// Stores a recipe's array part and fields into `table`. Nil values store
/// nothing.
pub(crate) fn fill_virtual(
    runtime: &mut Runtime,
    table: TableHandle,
    array: impl Iterator<Item = Value>,
    fields: impl Iterator<Item = (VirtualKey, Value)>,
) -> Result<(), RuntimeError> {
    for (position, value) in array.enumerate() {
        if value != Value::Nil {
            let index = i64::try_from(position + 1).unwrap_or(i64::MAX);
            let key = integer_key(runtime, index);
            runtime.table_set(table, &key, value)?;
        }
    }
    for (key, value) in fields {
        if value != Value::Nil {
            let key = virtual_key_value(runtime, &key);
            runtime.table_set(table, &key, value)?;
        }
    }
    Ok(())
}

/// The key of a recipe field as [`fill_virtual_keyed`] takes it: the
/// recipe's key, or the string made of it beforehand.
pub(crate) enum VirtualFieldKey<'a> {
    Recipe(&'a VirtualKey),
    Made(&'a LuaString),
}

/// [`fill_virtual`] with each field's key as a [`VirtualFieldKey`], so a
/// plan that kept the strings of its keys makes none here. The stores and
/// their order are those of [`fill_virtual`].
pub(crate) fn fill_virtual_keyed<'a>(
    runtime: &mut Runtime,
    table: TableHandle,
    array: impl Iterator<Item = Value>,
    fields: impl Iterator<Item = (VirtualFieldKey<'a>, Value)>,
) -> Result<(), RuntimeError> {
    for (position, value) in array.enumerate() {
        if value != Value::Nil {
            let index = i64::try_from(position + 1).unwrap_or(i64::MAX);
            let key = integer_key(runtime, index);
            runtime.table_set(table, &key, value)?;
        }
    }
    for (key, value) in fields {
        if value != Value::Nil {
            let key = match key {
                VirtualFieldKey::Recipe(key) => virtual_key_value(runtime, key),
                VirtualFieldKey::Made(string) => Value::String(string.clone()),
            };
            runtime.table_set(table, &key, value)?;
        }
    }
    Ok(())
}

/// A value a recipe stores once resolved: a VM value, or the table another
/// recipe of the snapshot builds (by index into the snapshot's recipes).
enum ResolvedItem {
    Value(Value),
    Virtual(usize),
}

/// A recipe with its values resolved: its index, array, fields and metatable.
type ResolvedRecipe = (
    usize,
    Vec<ResolvedItem>,
    Vec<(VirtualKey, ResolvedItem)>,
    Option<Value>,
);

/// [`resolve_recipe_entry`] that also takes a table built inside the table:
/// the recipe `id` of the same snapshot.
fn resolve_item(
    entry: &RegisterEntry,
    values: &(impl ExitValues + ?Sized),
    bank: &[Value],
    roots: &[Value],
    count: usize,
) -> Result<ResolvedItem, DeoptError> {
    match entry {
        RegisterEntry::Virtual(id) if id.index() < count => Ok(ResolvedItem::Virtual(id.index())),
        RegisterEntry::Virtual(id) => Err(DeoptError::MissingVirtual(*id)),
        other => resolve_recipe_entry(other, values, bank, roots).map(ResolvedItem::Value),
    }
}

/// Builds the virtual tables the frames of `snapshot` name, for the generic
/// restore: all of them are allocated first, then each is filled, and nothing
/// else runs in between. No safepoint falls inside, so the collector never
/// sees a partial table. The result is indexed by [`VirtualId`]; a recipe no
/// register names is not built, but one a built recipe holds is. A recipe
/// with `phantom_entries` set gives its phantom charge back, since the
/// allocation now counts.
///
/// # Errors
///
/// A [`DeoptError`] when a value cannot be resolved or an allocation fails.
/// Tables made before the failure stay in the arena, unreachable, for the
/// next collection.
pub(crate) fn materialize_virtuals(
    runtime: &mut Runtime,
    snapshot: &Snapshot,
    values: &(impl ExitValues + ?Sized),
    bank: &[Value],
    roots: &[Value],
) -> Result<Vec<Option<TableHandle>>, DeoptError> {
    let mut handles: Vec<Option<TableHandle>> = vec![None; snapshot.virtuals.len()];
    if snapshot.virtuals.is_empty() {
        return Ok(handles);
    }
    let mut named = vec![false; snapshot.virtuals.len()];
    for entry in snapshot.frames.iter().flat_map(|frame| &frame.registers) {
        if let RegisterEntry::Virtual(id) = entry {
            *named
                .get_mut(id.index())
                .ok_or(DeoptError::MissingVirtual(*id))? = true;
        }
    }
    // A table a built table holds is built too, whatever names it.
    let mut changed = true;
    while changed {
        changed = false;
        for (index, table) in snapshot.virtuals.iter().enumerate() {
            if !named[index] {
                continue;
            }
            for entry in table.entries() {
                if let RegisterEntry::Virtual(id) = entry {
                    let inner = named
                        .get_mut(id.index())
                        .ok_or(DeoptError::MissingVirtual(*id))?;
                    changed |= !*inner;
                    *inner = true;
                }
            }
        }
    }
    // Resolve every value before the first allocation.
    let count = snapshot.virtuals.len();
    let mut resolved: Vec<ResolvedRecipe> = Vec::new();
    for (index, table) in snapshot.virtuals.iter().enumerate() {
        if !named[index] {
            continue;
        }
        let array = table
            .array
            .iter()
            .map(|entry| resolve_item(entry, values, bank, roots, count))
            .collect::<Result<Vec<_>, _>>()?;
        let fields = table
            .fields
            .iter()
            .map(|(key, entry)| {
                Ok((
                    key.clone(),
                    resolve_item(entry, values, bank, roots, count)?,
                ))
            })
            .collect::<Result<Vec<_>, DeoptError>>()?;
        let metatable = table
            .metatable
            .as_ref()
            .map(|entry| resolve_recipe_entry(entry, values, bank, roots))
            .transpose()?;
        if metatable
            .as_ref()
            .is_some_and(|value| !matches!(value, Value::Table(_)))
        {
            return Err(DeoptError::Materialize(
                "a virtual table's metatable is a table".to_owned(),
            ));
        }
        resolved.push((index, array, fields, metatable));
    }
    let failed = |error: RuntimeError| DeoptError::Materialize(error.to_string());
    for (index, ..) in &resolved {
        let table: &VirtualTable = &snapshot.virtuals[*index];
        let shape = virtual_shape(&table.alloc);
        handles[*index] = Some(allocate_virtual(runtime, shape.as_ref()).map_err(failed)?);
        if let Some(entries) = table.phantom_entries {
            runtime.refund_elided_table(entries);
        }
    }
    let item_value = |item: ResolvedItem, handles: &[Option<TableHandle>]| match item {
        ResolvedItem::Value(value) => Ok(value),
        ResolvedItem::Virtual(inner) => handles[inner].map(Value::Table).ok_or_else(|| {
            DeoptError::Materialize("a table inside a table was not built".to_owned())
        }),
    };
    for (index, array, fields, metatable) in resolved {
        let handle = handles[index].expect("allocated above");
        let array = array
            .into_iter()
            .map(|item| item_value(item, &handles))
            .collect::<Result<Vec<_>, _>>()?;
        let fields = fields
            .into_iter()
            .map(|(key, item)| Ok((key, item_value(item, &handles)?)))
            .collect::<Result<Vec<_>, DeoptError>>()?;
        fill_virtual(runtime, handle, array.into_iter(), fields.into_iter()).map_err(failed)?;
        // After the contents, as the interpreter's `setmetatable` came after
        // the constructor. The trace checked that the metatable asks for no
        // finalizer and no weak mode.
        if let Some(Value::Table(metatable)) = metatable {
            runtime
                .set_table_metatable(handle, Some(metatable))
                .map_err(failed)?;
        }
    }
    Ok(handles)
}

fn check_pcs(code_len: usize, resume: ResumePoint) -> Result<(), DeoptError> {
    for pc in std::iter::once(resume.pc).chain(resume.last_interpreted_pc) {
        if pc >= code_len {
            return Err(DeoptError::PcOutOfBounds(pc));
        }
    }
    Ok(())
}

/// Restores a root-frame `snapshot` onto `frame`, with its whole window.
///
/// `values` supplies the SSA values the snapshot's `Value` entries name,
/// `bank` resolves table and closure lanes, and `roots` backs `Root` entries.
/// `externals` says what the window registers beyond the frame's own are.
/// `Unchanged` entries leave the register alone, `Const` entries write the
/// constant. On success the frame's `pc` and `last_interpreted_pc` follow the
/// snapshot's phase. `virtuals` holds the tables [`materialize_virtuals`]
/// built for the snapshot's `Virtual` entries; an entry with none is an error.
/// The cell writes the upvalue slots call for come back in
/// [`RootRestore::upvalues`]. A
/// snapshot inside an inlined call restores through [`restore_snapshot`].
///
/// # Errors
///
/// Returns a [`DeoptError`] without touching the frame when the snapshot does
/// not fit it or an entry cannot be resolved.
pub(crate) fn restore_window(
    frame: &mut FrameMut<'_>,
    snapshot: &Snapshot,
    values: &(impl ExitValues + ?Sized),
    bank: &[Value],
    roots: &[Value],
    externals: &[ExternalSlot],
    virtuals: &[Option<TableHandle>],
) -> Result<RootRestore, DeoptError> {
    let [root] = snapshot.frames.as_slice() else {
        return Err(DeoptError::NotRootOnly {
            frames: snapshot.frames.len(),
        });
    };
    // A later activation of the root frame needs the stack to resize the
    // frame: see `restore_snapshot`.
    if root.frame != FrameId(0) {
        return Err(DeoptError::NotRootFrame(root.frame));
    }
    let real = frame.registers.len();
    if root.registers.len() != real + externals.len() {
        return Err(DeoptError::RegisterCount {
            snapshot: root.registers.len(),
            frame: real + externals.len(),
        });
    }
    let resume = snapshot.resume_point();
    check_pcs(frame.chunk.code.len(), resume)?;
    // Resolve everything first so a failure writes nothing.
    let mut resolved = resolve_registers(
        &root.registers,
        &snapshot.numbers,
        values,
        bank,
        roots,
        virtuals,
    )?;
    let upvalues = cell_writes(externals, resolved.drain(real..))?;
    for (register, source) in frame.registers.iter_mut().zip(resolved) {
        if let Source::Value(value) = source {
            *register = value;
        }
    }
    frame.pc = resume.pc;
    frame.last_interpreted_pc = resume.last_interpreted_pc.into();
    Ok(RootRestore { resume, upvalues })
}

/// [`restore_window`] for a trace with no table or closure lanes, no upvalue
/// or global slots: the registers of one root frame and nothing else.
///
/// # Errors
///
/// As [`restore_window`].
pub(crate) fn restore_root_snapshot(
    frame: &mut FrameMut<'_>,
    snapshot: &Snapshot,
    values: &(impl ExitValues + ?Sized),
    roots: &[Value],
) -> Result<ResumePoint, DeoptError> {
    restore_window(frame, snapshot, values, &[], roots, &[], &[]).map(|restore| restore.resume)
}

/// Pushes the activation record the interpreter's own call of `closure` would
/// have made, returning to `call`'s result window, with every register nil.
/// Returns the new frame's index, or `None` when the closure no longer exists.
pub(crate) fn push_inlined_frame(
    frames: &mut CallStack,
    runtime: &Runtime,
    closure: ClosureHandle,
    call: CallReturn,
) -> Option<usize> {
    let parts = runtime.closure_call_parts(closure)?;
    // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
    #[allow(unsafe_code)]
    let index = unsafe {
        frames.push_activation(
            parts.chunk,
            parts.cells,
            parts.environment,
            closure,
            call,
            ActivationArguments::None,
        )
    };
    Some(index)
}

/// [`push_inlined_frame`] for a frame `protected` says a `pcall` or `xpcall`
/// made, or `entry` says a metamethod instruction entered: it returns to the
/// target the interpreter's own frame does, so an error the interpreter
/// raises later in it is caught there and its return does what the
/// instruction's does. `None` when the closure no longer exists or a handler
/// is not in `bank`.
pub(crate) fn push_protected_frame(
    frames: &mut CallStack,
    runtime: &Runtime,
    bank: &[Value],
    closure: ClosureHandle,
    call: CallReturn,
    (protected, entry): (Protected, Entry),
) -> Option<usize> {
    let target = match (protected, entry) {
        (Protected::No, Entry::Call) => return push_inlined_frame(frames, runtime, closure, call),
        (Protected::Pcall, _) => ReturnTarget::Protected(call),
        (Protected::Xpcall { handler }, _) => {
            ReturnTarget::ExtendedProtectedTarget(Box::new(ExtendedProtectedCall {
                target: call,
                handler: bank.get(handler as usize)?.clone(),
                suppress_hooks: false,
                handler_calls: 0,
                pending_unwind: None,
            }))
        }
        // The values the continuation holds are what the instruction held;
        // a return reads only the register to write, and the collector
        // finds the operands in the caller's registers.
        (Protected::No, Entry::Index) => ReturnTarget::IndexResult(call.base),
        (Protected::No, Entry::NewIndex) => ReturnTarget::SetMetamethod,
        // The operands the chain has not folded in are the registers of the
        // caller from `first` on, which the instruction leaves alone. The
        // accumulator is the handler's result, which the return supplies.
        (
            Protected::No,
            Entry::Concat {
                first,
                last,
                pending,
            },
        ) => {
            let caller = frames.last()?;
            let from = usize::from(first);
            let remaining = caller
                .registers
                .get(from..from + usize::from(pending))?
                .to_vec();
            ReturnTarget::ConcatRange(Box::new(ConcatRangeContinuation {
                destination: call.base,
                first,
                last,
                remaining,
                accumulator: Value::Nil,
            }))
        }
        // The continuation holds the argument only for the collector, and
        // the caller's registers still hold it.
        (Protected::No, Entry::ToString) => {
            ReturnTarget::ToString(Box::new(ToStringContinuation {
                completion: ToStringCompletion::Direct(call),
                value: Value::Nil,
            }))
        }
        (Protected::No, Entry::Len | Entry::Operator) => {
            ReturnTarget::Scalar(Box::new(ScalarContinuation::direct(call.base)))
        }
        (Protected::No, Entry::Compare { invert }) => ReturnTarget::Scalar(Box::new(
            ScalarContinuation::comparison(call.base, invert, Value::Nil, Value::Nil),
        )),
        // A caller frame is the interpreter's own, no exit builds it.
        (Protected::No, Entry::Caller { .. }) => return None,
        (Protected::No, Entry::Branch { expected, invert }) => {
            let mut continuation = ScalarContinuation::branching(expected, Value::Nil, Value::Nil);
            continuation.invert = invert;
            ReturnTarget::Scalar(Box::new(continuation))
        }
    };
    let parts = runtime.closure_call_parts(closure)?;
    // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`.
    #[allow(unsafe_code)]
    let index = unsafe {
        frames.push_activation_to(
            parts.chunk,
            parts.cells,
            parts.environment,
            closure,
            target,
            ActivationArguments::None,
        )
    };
    Some(index)
}

/// Gives the rebuilt frame of an inlined call what its tail calls left on the
/// interpreter's frame: their count, and the call site it is named after (the
/// last tail call's instruction, in the chunk of the closure that made it).
pub(crate) fn restore_tail_state(
    frame: &mut FrameMut<'_>,
    (tail_calls, tail_site): (u32, Option<(u32, usize)>),
    bank: &[Value],
    runtime: &Runtime,
) {
    if tail_calls != frame.tail_calls || tail_site.is_some() {
        frame.clear_plain_mark();
    }
    frame.tail_calls = tail_calls;
    if let Some((lane, pc)) = tail_site
        && let Some(handle) = pinned_closure(bank, lane)
        && let Some(parts) = runtime.closure_call_parts(handle)
    {
        frame.call_site = crate::vm::debug_info::call_site_at(parts.chunk, pc + 1)
            .map(crate::vm::FrameSite::boxed);
    }
}

/// What the root frame's tail calls left on it at a snapshot: the closure a
/// later activation runs, and the tail call it is named after.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RootTail {
    /// The pinned closure of a later activation, or `None` while the root
    /// frame runs the closure it was entered with.
    pub(crate) closure: Option<u32>,
    pub(crate) site: TailSite,
}

impl RootTail {
    /// The state of a snapshot whose first frame is `root`, for a trace that
    /// loops through tail calls of the root frame. `None` for any other
    /// trace. Before the first tail call of an iteration, the last one was
    /// the previous iteration's closing one.
    pub(crate) fn of(ir: &TraceIr, root: FrameId) -> Option<Self> {
        let closing = ir.loop_tail?;
        let activation = ir.frames.get(root.index())?.activation;
        Some(match activation {
            Some(activation) => Self {
                closure: Some(activation.closure),
                site: activation.site,
            },
            None => Self {
                closure: None,
                site: closing,
            },
        })
    }
}

/// Makes the top frame at `index` run `closure` from its pc 0, as the
/// interpreter's in-place tail call does (`Vm::try_fixed_tail_call`): the
/// closure's chunk, cells and environment, cleared interior state, and the
/// chunk's register count, all nil. `None` when the closure no longer exists.
pub(crate) fn replace_root_activation(
    frames: &mut CallStack,
    index: usize,
    runtime: &Runtime,
    closure: ClosureHandle,
) -> Option<()> {
    debug_assert_eq!(index + 1, frames.len(), "the root frame is the top frame");
    let parts = runtime.closure_call_parts(closure)?;
    frames.resize_top(usize::from(parts.chunk.register_count));
    let mut frame = frames.at_mut(index);
    frame.clear_plain_mark();
    // SAFETY: `parts` are `closure`'s own, and the frame roots `closure`
    // from here on, as the interpreter's in-place tail call does.
    #[allow(unsafe_code)]
    unsafe {
        frame.chunk = FrameChunk::borrowed(parts.chunk);
        frame.upvalues = CaptureCells::borrow_shared(parts.cells);
    }
    frame.closure = Some(closure);
    frame.environment = parts.environment;
    frame.shape = parts.shape;
    frame.reset_extras();
    frame.pc = 0;
    frame.last_interpreted_pc = crate::vm::frame::LastPc::NONE;
    Some(())
}

/// Leaves on the top frame at `index` what the root frame's tail calls in
/// the run did, given the tail call count `count` the trace reached (see the
/// module docs). Registers, `pc` and `last_interpreted_pc` are the caller's
/// to write afterwards.
///
/// # Errors
///
/// [`DeoptError::MissingClosure`] when a pinned closure is not in the bank or
/// no longer exists. Nothing is written then.
pub(crate) fn apply_root_tail(
    frames: &mut CallStack,
    index: usize,
    tail: RootTail,
    count: i64,
    bank: &[Value],
    runtime: &Runtime,
) -> Result<(), DeoptError> {
    let pinned = |lane: u32| {
        pinned_closure(bank, lane)
            .filter(|handle| runtime.closure_call_parts(*handle).is_some())
            .ok_or(DeoptError::MissingClosure(lane))
    };
    let frame = frames.at(index);
    let changed = count != i64::from(frame.tail_calls);
    debug_assert!(
        changed || tail.closure.is_none(),
        "an activation counts its call"
    );
    // The name comes from the chunk of the closure that made the call: a
    // pinned one, or the one the frame runs now (before any replacement).
    let call_site = if changed {
        let site = match tail.site.closure {
            Some(lane) => {
                let parts = runtime
                    .closure_call_parts(pinned(lane)?)
                    .ok_or(DeoptError::MissingClosure(lane))?;
                crate::vm::debug_info::call_site_at(parts.chunk, tail.site.pc + 1)
            }
            None => crate::vm::debug_info::call_site_at(&frame.chunk, tail.site.pc + 1),
        };
        Some(site)
    } else {
        None
    };
    let replacement = tail.closure.map(pinned).transpose()?;
    if let Some(handle) = replacement {
        replace_root_activation(frames, index, runtime, handle)
            .ok_or(DeoptError::MissingClosure(tail.closure.unwrap_or_default()))?;
    }
    if let Some(site) = call_site {
        let mut frame = frames.at_mut(index);
        frame.clear_plain_mark();
        frame.reset_extras();
        match (site, &mut frame.call_site) {
            (Some(site), Some(held)) => **held = crate::vm::FrameSite::Named(site),
            (site, held) => *held = site.map(crate::vm::FrameSite::boxed),
        }
        frame.tail_calls = u32::try_from(count).unwrap_or(u32::MAX);
    }
    Ok(())
}

/// Performs the cell `writes` of a restore: [`CellTarget::Frame`] through
/// `cells`, the root frame's, and [`CellTarget::Closure`] through the pinned
/// closure in `bank`. A cell that cannot be found is skipped.
fn write_cells(
    cells: &CaptureCells,
    writes: &[(CellTarget, Value)],
    bank: &[Value],
    runtime: &mut Runtime,
) {
    for (target, value) in writes {
        let cell = match *target {
            CellTarget::Frame(index) => cells.get(usize::from(index)).copied(),
            CellTarget::Closure { lane, index } => pinned_closure(bank, lane)
                .and_then(|handle| runtime.closure_cell(handle, usize::from(index)).ok())
                .flatten(),
            // The caller's registers are written by `write_caller_registers`.
            CellTarget::Caller(_) => None,
        };
        if let Some(cell) = cell {
            let written = runtime.cell_set(cell, value.clone());
            debug_assert!(written.is_ok(), "a restored trace's cells exist");
        }
    }
}

/// Performs the writes of a restore that name registers of the frame below
/// the root frame at `root_index`: the caller's.
fn write_caller_registers(
    frames: &mut CallStack,
    root_index: usize,
    writes: &[(CellTarget, Value)],
) {
    for (target, value) in writes {
        if let CellTarget::Caller(register) = *target
            && let Some(below) = root_index.checked_sub(1)
            && let Some(slot) = frames
                .at_mut(below)
                .registers
                .get_mut(usize::from(register))
        {
            *slot = value.clone();
        }
    }
}

/// The closure pinned as lane `lane`, from the run's bank.
pub(crate) fn pinned_closure(bank: &[Value], lane: u32) -> Option<ClosureHandle> {
    match bank.get(lane as usize)? {
        Value::Closure(handle) => Some(*handle),
        _ => None,
    }
}

/// What [`restore_snapshot`] leaves for the caller to do.
#[derive(Debug)]
pub(crate) struct Restore {
    /// Where the innermost frame resumes.
    pub(crate) resume: ResumePoint,
    /// The cell writes, which [`restore_snapshot`] has already made.
    pub(crate) upvalues: Vec<(CellTarget, Value)>,
}

/// Restores `snapshot`, which may sit inside inlined calls, onto the frame
/// stack: the root frame at `root_index` is rewritten, and each inlined frame
/// is pushed above it in call order.
///
/// `ir` is the trace the snapshot belongs to. Its frames say how each
/// callee was entered, and `bank` holds the pinned closures. Unlike
/// [`restore_window`], this writes the cells itself: a [`CellTarget::Frame`]
/// write names a cell of the closure the root frame was entered with, which
/// a root tail call replaces, so the writes go first, as the native plan's
/// `apply_cells` runs before its `apply_root_tail`. [`Restore::upvalues`]
/// lists them.
///
/// # Errors
///
/// Returns a [`DeoptError`] without touching the stack when the snapshot does
/// not fit it or an entry cannot be resolved.
#[allow(clippy::too_many_arguments)]
pub(crate) fn restore_snapshot(
    frames: &mut CallStack,
    root_index: usize,
    runtime: &mut Runtime,
    ir: &TraceIr,
    snapshot: &Snapshot,
    values: &(impl ExitValues + ?Sized),
    bank: &[Value],
    roots: &[Value],
    externals: &[ExternalSlot],
    virtuals: &[Option<TableHandle>],
) -> Result<Restore, DeoptError> {
    let (root, inner) = snapshot
        .frames
        .split_first()
        .expect("a snapshot has a frame");
    if !ir
        .frames
        .get(root.frame.index())
        .is_some_and(crate::jit_ir::trace::TraceFrameInfo::is_root)
    {
        return Err(DeoptError::NotRootFrame(root.frame));
    }
    // The window is the root frame's at entry, whichever activation runs.
    let real = usize::from(ir.frames[0].register_count).saturating_sub(externals.len());
    if root.registers.len() != real + externals.len()
        || (ir.loop_tail.is_none() && frames.at(root_index).registers.len() != real)
    {
        return Err(DeoptError::RegisterCount {
            snapshot: root.registers.len(),
            frame: real + externals.len(),
        });
    }
    let root_tail = RootTail::of(ir, root.frame);
    let root_code_len = match root_tail.and_then(|tail| tail.closure) {
        Some(_) => ir.frames[root.frame.index()].code_len,
        None => frames.at(root_index).chunk.code.len(),
    };
    // The frame that resumes: the innermost one. The frames outside it stand
    // at the call that made the next one.
    let resume = snapshot.resume_point();
    let call_of = |frame: FrameId| {
        ir.frames
            .get(frame.index())
            .and_then(|info| info.inlined.map(|call| (info, call)))
            .ok_or(DeoptError::BadInlinedFrame(frame))
    };
    let mut resolved_root = resolve_registers(
        &root.registers,
        &snapshot.numbers,
        values,
        bank,
        roots,
        virtuals,
    )?;
    // The tail call count the run reached, which `apply_root_tail` leaves on
    // the frame.
    let count = match externals
        .iter()
        .position(|slot| *slot == ExternalSlot::TailCalls)
        .map(|position| &resolved_root[real + position])
    {
        Some(Source::Value(Value::Integer(count))) => Some(*count),
        Some(Source::Keep) | None => None,
        Some(Source::Value(_)) => return Err(DeoptError::TailCount),
    };
    let upvalues = cell_writes(externals, resolved_root.drain(real..))?;
    let mut planned = Vec::with_capacity(inner.len());
    let mut lanes = Vec::with_capacity(inner.len());
    for (position, frame) in inner.iter().enumerate() {
        let (info, call) = call_of(frame.frame)?;
        if frame.registers.len() != usize::from(info.register_count) {
            return Err(DeoptError::BadInlinedFrame(frame.frame));
        }
        let mut registers =
            resolve_registers(&frame.registers, &[], values, bank, roots, virtuals)?;
        // The root frame's caller is the interpreter's frame, which the
        // restore writes and resumes: it needs no closure.
        let handle = if matches!(call.entry, Entry::Caller { .. }) {
            None
        // A frame that runs the closure of this run carries it in a register
        // after the chunk's.
        } else if call.dynamic {
            match registers.pop() {
                Some(Source::Value(Value::Closure(handle))) => Some(handle),
                _ => return Err(DeoptError::MissingClosure(call.closure)),
            }
        } else {
            Some(
                pinned_closure(bank, call.closure)
                    .ok_or(DeoptError::MissingClosure(call.closure))?,
            )
        };
        // Stopped at its own call, or resuming if it is the innermost.
        let at = match inner.get(position + 1) {
            Some(next) => {
                let (_, next_call) = call_of(next.frame)?;
                ResumePoint {
                    pc: next_call.call_pc + 1,
                    last_interpreted_pc: Some(next_call.call_pc),
                }
            }
            None => resume,
        };
        check_pcs(info.code_len, at)?;
        planned.push((handle, call, registers, at));
        lanes.push(call.closure);
    }
    let root_at = match inner.first() {
        Some(first) => {
            let (_, call) = call_of(first.frame)?;
            ResumePoint {
                pc: call.call_pc + 1,
                last_interpreted_pc: Some(call.call_pc),
            }
        }
        None => resume,
    };
    // The root frame goes when its caller is the first frame above it.
    if !inner.first().is_some_and(|first| {
        call_of(first.frame).is_ok_and(|(_, call)| matches!(call.entry, Entry::Caller { .. }))
    }) {
        check_pcs(root_code_len, root_at)?;
    }
    // Every closure must exist before the first write.
    for ((handle, ..), lane) in planned.iter().zip(&lanes) {
        if let Some(handle) = handle
            && runtime.closure_call_parts(*handle).is_none()
        {
            return Err(DeoptError::MissingClosure(*lane));
        }
    }
    if let Some(lane) = root_tail.and_then(|tail| tail.closure)
        && pinned_closure(bank, lane)
            .and_then(|handle| runtime.closure_call_parts(handle))
            .is_none()
    {
        return Err(DeoptError::MissingClosure(lane));
    }

    // Before the root tail state replaces the frame's cells.
    write_cells(
        &frames.at(root_index).frame.upvalues,
        &upvalues,
        bank,
        runtime,
    );
    write_caller_registers(frames, root_index, &upvalues);
    match (root_tail, count) {
        (Some(tail), Some(count)) => {
            apply_root_tail(frames, root_index, tail, count, bank, runtime)?;
        }
        // An activation always follows a counted tail call.
        (
            Some(RootTail {
                closure: Some(_), ..
            }),
            None,
        ) => return Err(DeoptError::TailCount),
        _ => {}
    }
    let mut frame = frames.at_mut(root_index);
    for (register, source) in frame.registers.iter_mut().zip(resolved_root) {
        if let Source::Value(value) = source {
            *register = value;
        }
    }
    frame.pc = root_at.pc;
    frame.last_interpreted_pc = root_at.last_interpreted_pc.into();
    for (handle, call, registers, at) in planned {
        let Some(handle) = handle else {
            // The root frame has returned: its caller resumes, and the root
            // frame goes as a `Return` would take it.
            let below = frames
                .len()
                .checked_sub(2)
                .ok_or(DeoptError::BadInlinedFrame(root.frame))?;
            {
                let mut caller = frames.at_mut(below);
                for (register, source) in caller.registers.iter_mut().zip(registers) {
                    if let Source::Value(value) = source {
                        *register = value;
                    }
                }
                caller.pc = at.pc;
                caller.last_interpreted_pc = at.last_interpreted_pc.into();
                caller.clear_open_results();
            }
            frames.retire_top();
            continue;
        };
        let index = push_protected_frame(
            frames,
            runtime,
            bank,
            handle,
            CallReturn {
                base: call.result_base,
                results: call.results,
            },
            (call.protected, call.entry),
        )
        .ok_or(DeoptError::MissingClosure(call.closure))?;
        let mut pushed = frames.at_mut(index);
        for (register, source) in pushed.registers.iter_mut().zip(registers) {
            if let Source::Value(value) = source {
                *register = value;
            }
        }
        pushed.pc = at.pc;
        pushed.last_interpreted_pc = at.last_interpreted_pc.into();
        restore_tail_state(
            &mut pushed,
            (call.tail_calls, call.tail_site),
            bank,
            runtime,
        );
    }
    Ok(Restore { resume, upvalues })
}
