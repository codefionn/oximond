//! The typed exit ABI of native traces.
//!
//! Native trace code never rebuilds guest frames. Entering, it reads typed
//! values from an input array. Leaving, an out-of-line stub per guard stores
//! the snapshot's live SSA values into a fixed exit area and returns the
//! index of an [`ExitDescriptor`]. The descriptor says which snapshot the
//! exit restores and where each value sits, so Rust can rebuild guest values
//! with their exact types: integer bits stay integers and float bits stay
//! floats. There are no uniform tagged lanes and nothing goes through `f64`
//! unless the value is a float.
//!
//! The exit area holds one 8-byte slot per spilled value: float bits, integer
//! bits, a boolean as 0 or 1, or the lane id of a table, closure or string. Nil
//! values need no slot.
//!
//! Table storage is the one place native code touches the heap. The VM
//! resolves a table's storage at entry (see
//! [`oximond::jit_ir::trace::TableBinding`]) and appends one [`TraceInputSlot`]
//! per binding after the entry inputs: for an array binding `bits` is the
//! address of slot 1 and `tag` the slot count, for a field binding `bits` is
//! the address of the slot's `Value`, and for a length binding `bits` holds
//! the length as integer or float bits. The addresses stay valid for the whole
//! run, because a run mutates table storage only by overwriting a payload with
//! a value of the same type, or through a table set the VM makes for it
//! ([`TraceTables::set`]). A set may grow the table, and rewrites the binding
//! slots of its arrays, which the code reads again after it. A tree that sets
//! binds no other storage a set could move.
//!
//! Tables and closures never cross as pointers or handle bits. The VM gives
//! each distinct opaque value an entry reads a lane id (an index into its
//! per-run root bank), native code moves and compares the ids, and an exit
//! hands them back for the VM to resolve through the same bank. Two lanes
//! are equal exactly when the objects are the same.

use crate::jit_ir::trace::{
    BindingKind, HostFunction, MAX_HELPER_RESULTS, MAX_HOST_ARGS, SnapshotId, TraceScalar,
    TraceType, ValueId,
};
use crate::runtime::{NativeBytes, NativeValueArray, NativeValueSlot, Value};

/// The top bit of a probe binding's mask word: native code must not store
/// through the index ([`TraceBinding::probe`]).
pub const PROBE_READ_ONLY: i64 = i64::MIN;
/// The next bit of a probe binding's mask word: native code must not delete
/// or revive an entry through the index, which leaves the table's live-entry
/// counter, whose address the slot after the index holds, to the VM.
pub const PROBE_NO_TOMBSTONES: i64 = 1 << 62;
/// Entry tag of a nil register.
pub const TRACE_TAG_NIL: i64 = 0;
/// Entry tag of a boolean register.
pub const TRACE_TAG_BOOLEAN: i64 = 1;
/// Entry tag of an integer register.
pub const TRACE_TAG_INTEGER: i64 = 2;
/// Entry tag of a float register.
pub const TRACE_TAG_FLOAT: i64 = 3;
/// Entry tag of any other register (userdata, native function, ...).
/// No trace input accepts it.
pub const TRACE_TAG_OTHER: i64 = 4;
/// Entry tag of a table register. The slot's bits are its lane id.
pub const TRACE_TAG_TABLE: i64 = 5;
/// Entry tag of a Lua closure register. The slot's bits are its lane id.
pub const TRACE_TAG_CLOSURE: i64 = 6;
/// Entry tag of a string register. The slot's bits are its lane id.
pub const TRACE_TAG_STRING: i64 = 7;
/// Entry tag of an `int64_t` cdata register. The slot's bits are its value.
pub const TRACE_TAG_INT64: i64 = 8;
/// Entry tag of a `uint64_t` cdata register. The slot's bits are its value.
pub const TRACE_TAG_UINT64: i64 = 9;

/// Key tag of a hash lookup whose key is a borrowed string
/// ([`TraceType::StringRef`]): the bits are the address of the table slot that
/// holds it. A lane string ([`TraceType::String`]) travels as
/// [`TRACE_TAG_STRING`] with its lane id.
pub const TRACE_TAG_BORROWED_STRING: i64 = 10;
/// Entry tag of a coroutine register. The slot's bits are its lane id.
pub const TRACE_TAG_THREAD: i64 = 11;

/// One register of the interpreter window as native code reads it at entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TraceInputSlot {
    pub(crate) tag: i64,
    /// Float bits, integer bits, or a boolean as 0 or 1.
    pub(crate) bits: i64,
}

/// The storage a trace's table binding resolved at entry, as the runtime
/// lent it.
///
/// Native code dereferences the addresses inside, so only the runtime's
/// views can make one: the fields are private and no constructor takes a raw
/// address outside tests. The views' lifetime contract applies to the run
/// that receives them: the caller holds exclusive access to the runtime and
/// mutates none of the tables between building the bindings and the end of
/// [`CompiledTrace::run_bound`](super::CompiledTrace::run_bound).
#[derive(Clone, Copy, Debug)]
pub struct TraceBinding(BindingWord);

#[derive(Clone, Copy, Debug)]
enum BindingWord {
    Array(NativeValueArray),
    Field(NativeValueSlot),
    Length(i64),
    Callee,
    Bytes(NativeBytes),
    /// A table's address and its flags, as the words (the flags of
    /// `NativeTableHeader::word` in the tag, the address in the bits).
    Header(i64, i64),
    /// The string index of a table's hash part, or none (zero slots): the
    /// words are the mask in the tag and the address of the slots in the bits.
    Probe(i64, i64),
    #[cfg(test)]
    RawArray(i64, i64),
    #[cfg(test)]
    RawField(i64),
}

impl TraceBinding {
    /// The array part of a table ([`BindingKind::Array`]).
    #[must_use]
    pub const fn array(view: NativeValueArray) -> Self {
        Self(BindingWord::Array(view))
    }

    /// One slot of a table ([`BindingKind::Field`]).
    #[must_use]
    pub const fn field(slot: NativeValueSlot) -> Self {
        Self(BindingWord::Field(slot))
    }

    /// A field the entry checked and the code never reads
    /// ([`BindingKind::Callee`], and the class of a [`BindingKind::MetaClass`],
    /// whose metatable identity is a constant of the code).
    #[must_use]
    pub const fn callee() -> Self {
        Self(BindingWord::Callee)
    }

    /// The bytes of a string ([`BindingKind::Bytes`]).
    #[must_use]
    pub const fn bytes(view: NativeBytes) -> Self {
        Self(BindingWord::Bytes(view))
    }

    /// A table's header ([`BindingKind::Header`]): the flags that say what
    /// native code may do to it in place (`NativeTableHeader::word`), and its
    /// address.
    #[must_use]
    pub fn header(flags: i64, address: usize) -> Self {
        Self(BindingWord::Header(
            flags,
            i64::try_from(address).unwrap_or(0),
        ))
    }

    /// The string index of a table's hash part ([`BindingKind::Hash`] or
    /// [`BindingKind::NoMetatable`]), which a lookup of a string key the
    /// trace holds probes in native code. The index is valid for the whole run
    /// when the trace moves no hash storage, the only trace that reads it.
    ///
    /// The mask word carries [`PROBE_READ_ONLY`] unless the table is one
    /// native code may overwrite a scalar of (see
    /// [`oximond::runtime::NativeStringProbe`]), and [`PROBE_NO_TOMBSTONES`]
    /// unless it may also delete and revive scalars, which `tombstones` says
    /// the run allows. A probe's own bucket index never reaches those bits,
    /// so a lookup is not told apart.
    #[must_use]
    pub fn probe(probe: crate::runtime::NativeStringProbe, tombstones: bool) -> Self {
        let mask = i64::try_from(probe.mask()).unwrap_or(0);
        let mut flags = 0;
        if !probe.writable() {
            flags |= PROBE_READ_ONLY;
        }
        if !(probe.writable() && probe.tombstones() && tombstones) {
            flags |= PROBE_NO_TOMBSTONES;
        }
        Self(BindingWord::Probe(
            mask | flags,
            i64::try_from(probe.slots()).unwrap_or(0),
        ))
    }

    /// A table border ([`BindingKind::Length`]) as an integer.
    #[must_use]
    pub const fn integer_length(length: i64) -> Self {
        Self(BindingWord::Length(length))
    }

    /// A table border ([`BindingKind::Length`]) as a float.
    #[must_use]
    pub fn float_length(length: f64) -> Self {
        Self(BindingWord::Length(i64::from_ne_bytes(
            length.to_bits().to_ne_bytes(),
        )))
    }

    /// Whether this binding supplies what `kind` resolves.
    pub(crate) fn matches(self, kind: &BindingKind) -> bool {
        match (self.0, kind) {
            (BindingWord::Array(_), BindingKind::Array { .. })
            | (
                BindingWord::Callee | BindingWord::Probe(..),
                BindingKind::NoMetatable | BindingKind::Hash,
            )
            | (BindingWord::Field(_), BindingKind::Field { .. })
            | (BindingWord::Length(_), BindingKind::Length { .. })
            | (
                BindingWord::Callee,
                BindingKind::Callee { .. }
                | BindingKind::StringField { .. }
                | BindingKind::MetaClass { .. },
            )
            | (BindingWord::Header(..), BindingKind::Header { .. })
            | (BindingWord::Bytes(_), BindingKind::Bytes { .. }) => true,
            #[cfg(test)]
            (BindingWord::RawArray(..), BindingKind::Array { .. })
            | (BindingWord::RawField(_), BindingKind::Field { .. }) => true,
            _ => false,
        }
    }

    /// The input slot native code reads.
    pub(crate) fn word(self) -> TraceInputSlot {
        match self.0 {
            BindingWord::Array(view) => TraceInputSlot {
                tag: i64::try_from(view.len()).unwrap_or(0),
                bits: view.as_ptr() as usize as i64,
            },
            BindingWord::Field(slot) => TraceInputSlot {
                tag: 0,
                bits: slot.as_ptr() as usize as i64,
            },
            BindingWord::Length(bits) => TraceInputSlot { tag: 0, bits },
            BindingWord::Callee => TraceInputSlot { tag: 0, bits: 0 },
            BindingWord::Bytes(view) => TraceInputSlot {
                tag: i64::try_from(view.len()).unwrap_or(0),
                bits: i64::try_from(view.as_ptr().addr()).unwrap_or(0),
            },
            BindingWord::Header(tag, bits) | BindingWord::Probe(tag, bits) => {
                TraceInputSlot { tag, bits }
            }
            #[cfg(test)]
            BindingWord::RawArray(tag, bits) => TraceInputSlot { tag, bits },
            #[cfg(test)]
            BindingWord::RawField(bits) => TraceInputSlot { tag: 0, bits },
        }
    }

    /// Test-only: an array over memory the test owns.
    #[cfg(test)]
    pub(crate) fn raw_array(len: usize, address: usize) -> Self {
        Self(BindingWord::RawArray(
            i64::try_from(len).unwrap(),
            address as i64,
        ))
    }

    /// Test-only: a field over memory the test owns.
    #[cfg(test)]
    pub(crate) fn raw_field(address: usize) -> Self {
        Self(BindingWord::RawField(address as i64))
    }
}

impl TraceInputSlot {
    /// The slot's tag.
    #[must_use]
    pub const fn tag(self) -> i64 {
        self.tag
    }

    /// The slot's payload.
    #[must_use]
    pub const fn bits(self) -> i64 {
        self.bits
    }

    /// A slot of the given tag and bits.
    #[must_use]
    pub const fn from_words(tag: i64, bits: i64) -> Self {
        Self { tag, bits }
    }

    #[must_use]
    pub const fn nil() -> Self {
        Self {
            tag: TRACE_TAG_NIL,
            bits: 0,
        }
    }

    #[must_use]
    pub const fn boolean(value: bool) -> Self {
        Self {
            tag: TRACE_TAG_BOOLEAN,
            bits: value as i64,
        }
    }

    #[must_use]
    pub const fn integer(value: i64) -> Self {
        Self {
            tag: TRACE_TAG_INTEGER,
            bits: value,
        }
    }

    #[must_use]
    pub fn float(value: f64) -> Self {
        Self {
            tag: TRACE_TAG_FLOAT,
            bits: i64::from_ne_bytes(value.to_bits().to_ne_bytes()),
        }
    }

    /// A register no trace input can accept.
    #[must_use]
    pub const fn other() -> Self {
        Self {
            tag: TRACE_TAG_OTHER,
            bits: 0,
        }
    }

    /// A table register, named by the lane id the VM gave it for this run.
    #[must_use]
    pub const fn table(lane: u64) -> Self {
        Self {
            tag: TRACE_TAG_TABLE,
            bits: lane.cast_signed(),
        }
    }

    /// A closure register, named by the lane id the VM gave it for this run.
    #[must_use]
    pub const fn closure(lane: u64) -> Self {
        Self {
            tag: TRACE_TAG_CLOSURE,
            bits: lane.cast_signed(),
        }
    }

    /// A coroutine register, named by the lane id the VM gave it for this run.
    #[must_use]
    pub const fn thread(lane: u64) -> Self {
        Self {
            tag: TRACE_TAG_THREAD,
            bits: lane.cast_signed(),
        }
    }

    /// A string register, named by the lane id the VM gave it for this run.
    #[must_use]
    pub const fn string(lane: u64) -> Self {
        Self {
            tag: TRACE_TAG_STRING,
            bits: lane.cast_signed(),
        }
    }

    /// An `int64_t` cdata register.
    #[must_use]
    pub const fn int64(value: i64) -> Self {
        Self {
            tag: TRACE_TAG_INT64,
            bits: value,
        }
    }

    /// A `uint64_t` cdata register.
    #[must_use]
    pub const fn uint64(value: u64) -> Self {
        Self {
            tag: TRACE_TAG_UINT64,
            bits: value.cast_signed(),
        }
    }

    /// Makes the slot the words of an array binding: the slot count and the
    /// address of slot 1, each written on its own.
    #[inline]
    pub fn set_array(&mut self, view: NativeValueArray) {
        self.tag = i64::try_from(view.len()).unwrap_or(0);
        self.bits = i64::try_from(view.as_ptr().addr()).unwrap_or(0);
    }

    /// The value a scalar slot (nil, boolean, integer or float) stands for.
    #[inline]
    #[must_use]
    pub fn scalar_value(self) -> Option<Value> {
        Some(match self.tag {
            TRACE_TAG_NIL => Value::Nil,
            TRACE_TAG_BOOLEAN => Value::Boolean(self.bits != 0),
            TRACE_TAG_INTEGER => Value::Integer(self.bits),
            TRACE_TAG_FLOAT => Value::Number(f64::from_bits(self.bits.cast_unsigned())),
            _ => return None,
        })
    }

    /// The slot for a scalar.
    #[must_use]
    pub fn from_scalar(scalar: TraceScalar) -> Self {
        match scalar {
            TraceScalar::Nil => Self::nil(),
            TraceScalar::Boolean(value) => Self::boolean(value),
            TraceScalar::Integer(value) => Self::integer(value),
            TraceScalar::Float(value) => Self::float(value),
            TraceScalar::Int64(value) => Self::int64(value),
            TraceScalar::UInt64(value) => Self::uint64(value),
            TraceScalar::Table(lane) => Self::table(lane),
            TraceScalar::Closure(lane) => Self::closure(lane),
            TraceScalar::Thread(lane) => Self::thread(lane),
            // No entry accepts a userdata: the tag no input requires.
            TraceScalar::Userdata(lane) => Self {
                tag: TRACE_TAG_OTHER,
                bits: lane.cast_signed(),
            },
            TraceScalar::String(lane) => Self::string(lane),
            // No entry accepts a borrow: the tag no input requires.
            TraceScalar::StringRef(bits) => Self {
                tag: TRACE_TAG_OTHER,
                bits: bits.cast_signed(),
            },
        }
    }
}

/// The tag a trace input of type `ty` requires.
pub(crate) const fn entry_tag(ty: TraceType) -> Option<i64> {
    match ty {
        TraceType::Nil => Some(TRACE_TAG_NIL),
        TraceType::Boolean => Some(TRACE_TAG_BOOLEAN),
        TraceType::Integer => Some(TRACE_TAG_INTEGER),
        TraceType::Float => Some(TRACE_TAG_FLOAT),
        TraceType::Int64 => Some(TRACE_TAG_INT64),
        TraceType::UInt64 => Some(TRACE_TAG_UINT64),
        TraceType::Table => Some(TRACE_TAG_TABLE),
        TraceType::Closure => Some(TRACE_TAG_CLOSURE),
        TraceType::Thread => Some(TRACE_TAG_THREAD),
        TraceType::String => Some(TRACE_TAG_STRING),
        // A borrow is made inside a run, never read at entry.
        TraceType::StringRef | TraceType::Userdata | TraceType::Any => None,
    }
}

/// Per-run state the VM lends to native code.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceControl {
    /// Loop-backs the trace may still take. Each one costs one unit. At zero
    /// the trace leaves through its poll exit. On return it holds what is
    /// left, so `initial - fuel` is the number of completed loop-backs.
    pub fuel: i64,
    /// For a counted trace ([`oximond::jit_ir::trace::TraceIr::counted`]): the
    /// count hook's budget, the number of counted guest instructions that
    /// may still run before the hook fires, plus one. On entry it is the
    /// budget at the loop header. On return it is the budget at the start of
    /// the iteration that exited, and the exit's
    /// [`ExitDescriptor::counted`] instructions of that iteration ran after
    /// it. A trace that is not counted leaves it alone.
    pub budget: i64,
    /// Side traces entered natively during the run: those inside the tree's
    /// code are counted only by code compiled with
    /// [`oximond::jit::TraceCompileOptions::count_links`], and the calls of
    /// sides compiled apart always. Native code and the run add to it, so the
    /// caller starts it at zero.
    pub links: i64,
    /// Table constructors the run ran without allocating
    /// ([`TraceOp::PhantomTable`](oximond::jit_ir::trace::TraceOp::PhantomTable)),
    /// and the bytes their phantom debt comes to. Native code adds to both,
    /// so the caller starts them at zero and charges the collector once the
    /// run has returned.
    pub elided_tables: i64,
    pub elided_bytes: i64,
    /// Fields native code stored into shaped tables the run made, in place
    /// (see [`oximond::jit_ir::trace::HelperId::TABLE_INIT`]). Native code
    /// adds to it, so the caller starts it at zero and charges the collector
    /// once the run has returned.
    pub initialized: i64,
    /// Whether a table the run made takes a reference into a field without
    /// the collector's write barrier (the host sets it to 1 when no cycle is
    /// in progress, and the run runs nothing that could start one). Zero
    /// sends every such store to the helper.
    pub barrier_idle: i64,
    /// Array slots the run appended to tables in place
    /// ([`oximond::jit_ir::trace::BindingKind::Header`]). Native code adds to
    /// it, so the caller starts it at zero and charges the collector once
    /// the run has returned.
    pub appended: i64,
    /// The address of the run's row storage, set by
    /// [`CompiledTrace::run_with_tables`](super::CompiledTrace::run_with_tables)
    /// for the duration of a run and zero otherwise. Code that reads fields
    /// of run-time tables indexes it by lane.
    pub(crate) rows: usize,
    /// The address of the run's table dispatcher, set and cleared with
    /// [`Self::rows`].
    pub(crate) tables: usize,
    /// What the last table set the code asked the VM for returned: 0 when it
    /// stored, -1 when the guard must leave. The call that makes the set
    /// preserves every register, so it hands its result back here.
    pub(crate) set_status: i64,
    /// The lane the last admission the code asked the VM for through its
    /// register-preserving stub returned (see `define_admit_table_stub` in the
    /// emitter), or -1.
    pub(crate) admit_lane: i64,
    /// What the last helper the code called through a register-preserving
    /// stub returned (see `helper_stub_arity` in the emitter). The stub
    /// hands its result back here.
    pub(crate) helper_status: i64,
    /// What the last hash lookup the code asked the VM for found, as an entry
    /// tag and bits (`TRACE_TAG_OTHER` when the value is not a scalar or
    /// the lookup could not run). Like [`Self::set_status`], the call hands
    /// its result back here.
    pub(crate) hash_result: TraceInputSlot,
    /// How many hash lookups the run asked the VM for, which native code did
    /// not answer by probing the string index of the table. The run adds to
    /// it when it ends, so the caller starts it at zero.
    pub hash_calls: i64,
    /// Sides compiled apart from their tree that the run called. The run adds
    /// to it, so the caller starts it at zero.
    pub child_calls: i64,
    /// The sides the root's own exits called from native code (see
    /// `emit_native_dispatch`), which are not in `child_calls`. The run adds
    /// to it, so the caller starts it at zero.
    pub native_calls: i64,
    /// The address of the byte count of each string lane of the run's bank
    /// (`u32` per lane, [`MAX_TRACE_STRING_LANES`] of them), set with
    /// [`Self::rows`].
    pub(crate) lengths: usize,
    /// The address of the first slot of the run's bank, one value (16 bytes)
    /// per lane, set with [`Self::rows`] from [`TraceTables::bank_address`]
    /// (zero for a host that gives none). The bank does not move during a
    /// run. Native code reads and writes the lanes of a loop that keeps its
    /// control value there (see
    /// [`oximond::jit_ir::trace::TraceIr::boxed_steps`]), where a lane holds a
    /// number the code can overwrite without any count to keep.
    pub(crate) bank: usize,
    /// The address of the trace's root strings, one value (16 bytes) per root
    /// slot, and how many there are, set with [`Self::rows`] from
    /// [`TraceTables::roots_address`] (zero for a host that gives none). A
    /// string lane of [`ROOT_STRING_LANE_BASE`] or above names root slot
    /// `lane - ROOT_STRING_LANE_BASE`. Native code compares strings of these
    /// slots in place (`string_equal_by_words` in `emit.rs`).
    ///
    /// [`ROOT_STRING_LANE_BASE`]: crate::jit_ir::trace::ROOT_STRING_LANE_BASE
    pub(crate) roots: usize,
    pub(crate) roots_len: usize,
    /// The function native code calls for a host call, and the context it
    /// takes first (`extern "C" fn(context, args, out, function, argc)`),
    /// set with [`Self::rows`].
    pub(crate) host_entry: usize,
    pub(crate) host_context: usize,
    /// The function the table set stub calls, with the same context
    /// (`extern "C" fn(context, words, binding, key tag, key bits, value tag,
    /// value bits) -> status`).
    pub(crate) set_entry: usize,
    /// The function native code calls to store a string or table into an
    /// array slot, with the same context (`extern "C" fn(context, binding,
    /// tag, bits, slot, replace) -> status`, see [`HostEntry::store`]).
    pub(crate) store_entry: usize,
    /// The arguments of the host call native code is about to make, and the
    /// value the call answered with, each as an entry tag and bits
    /// ([`oximond::jit_ir::trace::TraceOp::HostCall`]).
    pub(crate) host_args: [TraceInputSlot; MAX_HOST_ARGS],
    pub(crate) host_result: TraceInputSlot,
    /// Where a helper writes its results, one word each
    /// ([`oximond::jit_ir::trace::TraceOp::CallHelper`]).
    pub(crate) helper_out: [i64; MAX_HELPER_RESULTS],
    /// For a function trace: the activation an exit left from, counting from
    /// 0 for the one the VM entered (see
    /// [`oximond::jit::CompiledTrace::activation_stride`]). The exit sets it; the
    /// activations around it, unwinding, leave it as it is.
    pub depth: i64,
    /// For a function trace: the deepest activation a run may reach. A self
    /// call at this depth leaves before the call instead, and the
    /// interpreter makes it. The caller sizes the exit area for it.
    pub depth_cap: i64,
    /// For a function trace: the address of the run's input slots and of its
    /// exit area, which the entry function sets for the activations, so that
    /// neither travels as an argument of every self call. A loop trace that
    /// links natively has the run set `inputs` (the buffer the link writes
    /// into), so that its exits read it where they need it instead of the
    /// loop keeping it in a register.
    pub(crate) inputs: usize,
    pub(crate) area: usize,
    /// The address of the run's index of table lanes and the stamp word of
    /// an entry in it that holds a table with its row filled, set with
    /// [`Self::rows`] from [`TraceTables::lane_index`] (zero for a host that
    /// gives none). Native code looks a table up in it before it asks the
    /// dispatcher for the table's lane.
    pub(crate) lane_index: usize,
    pub(crate) lane_ready: u64,
    /// The address of the root tree's cells, one per exit, which hold the
    /// address of the side compiled apart from that exit (or zero). Set by
    /// the run. Native exits of the root call the side from the cell
    /// themselves (see `emit_native_exit`).
    pub(crate) cells: usize,
    /// The tree the run is of, for native code to call the Rust chain of
    /// sides a native call returned a side's exit to (see
    /// `trace_call_sides`). Set by the run.
    pub(crate) tree: usize,
    /// Set by native code when a side it called linked back to the root's
    /// loop and the code re-entered the loop itself. The run then writes
    /// the root's writebacks as a link left them.
    pub(crate) linked: i64,
    /// Set by native code when it already called the sides for the code it
    /// returns (a side declined, or the chain ended), so that the run does not
    /// call them again and count the same calls twice.
    pub(crate) followed: i64,
    /// A word native code of the MIR backend keeps a result in when a
    /// conditional call may replace it (the float remainder). It is scratch:
    /// what it holds after a run is not part of the run's result.
    pub(crate) scratch: i64,
    /// Not zero when the run has stopped giving array slots the key strings
    /// of the tables they are looked up in (see `trace_adopt_key` in
    /// `mir/lower/probe.rs`). Native code reads it and that call sets it.
    pub(crate) adopt_off: i64,
}

impl TraceControl {
    /// A control block with `fuel` loop-backs and no links counted yet.
    #[must_use]
    pub const fn new(fuel: i64) -> Self {
        Self {
            fuel,
            budget: 0,
            links: 0,
            elided_tables: 0,
            elided_bytes: 0,
            initialized: 0,
            barrier_idle: 0,
            appended: 0,
            rows: 0,
            tables: 0,
            set_status: 0,
            admit_lane: 0,
            helper_status: 0,
            hash_result: TraceInputSlot::nil(),
            hash_calls: 0,
            child_calls: 0,
            native_calls: 0,
            lengths: 0,
            bank: 0,
            roots: 0,
            roots_len: 0,
            host_entry: 0,
            host_context: 0,
            set_entry: 0,
            store_entry: 0,
            host_args: [TraceInputSlot::nil(); MAX_HOST_ARGS],
            host_result: TraceInputSlot::nil(),
            helper_out: [0; MAX_HELPER_RESULTS],
            depth: 0,
            depth_cap: 0,
            inputs: 0,
            area: 0,
            lane_index: 0,
            lane_ready: 0,
            cells: 0,
            tree: 0,
            linked: 0,
            followed: 0,
            scratch: 0,
            adopt_off: 0,
        }
    }
}

/// The places of a [`TraceTables::lane_index`] table, a power of two.
pub const TRACE_LANE_INDEX_PLACES: usize = 256;

/// Most tables, by lane, one run can read fields of.
pub const MAX_TRACE_TABLE_LANES: usize = 256;

/// Most tables, by lane, one run of a function trace can read fields of or
/// make. A recursion over a tree meets a table at every call, and each lane
/// costs a run the entry and exit the cap forces once it is spent.
pub const MAX_FUNCTION_TABLE_LANES: usize = 2048;

/// Upvalue cells of a closure a row holds an address for.
pub const CLOSURE_ROW_CELLS: usize = 8;

/// The words of the row of a closure lane: a word naming its prototype (zero
/// for a lane that is no Lua closure), then the address of the value in each
/// of its first [`CLOSURE_ROW_CELLS`] upvalue cells (zero past its last).
/// The row shares the storage of the table rows, so a trace that reads and
/// writes callee cells natively has rows at least this long.
pub const CLOSURE_ROW_WORDS: usize = 1 + CLOSURE_ROW_CELLS;

/// The words of each row of a run's space: the fields of a table, or a
/// closure's words, and for a `masked` trace the presence mask after them
/// (see [`seal_row`]). Zero when neither needs a row.
#[must_use]
pub const fn row_stride_of(fields: usize, closure_words: usize, masked: bool) -> usize {
    let words = if fields > closure_words {
        fields
    } else {
        closure_words
    };
    if words == 0 {
        0
    } else if masked {
        words + 1
    } else {
        words
    }
}

/// Writes the presence mask of `row`, its last word: bit `i` is set iff word
/// `i` is not zero, for the first 64 words. Native code checks a group of
/// fields of one table with one load of the mask and one compare, so the
/// mask must be sealed after every write to a row's words.
#[allow(clippy::cast_possible_truncation)]
#[inline]
pub fn seal_row(row: &mut [usize]) {
    let Some((mask, words)) = row.split_last_mut() else {
        return;
    };
    let mut bits: u64 = 0;
    for (index, word) in words.iter().take(64).enumerate() {
        bits |= u64::from(*word != 0) << index;
    }
    *mask = bits as usize;
}

/// Fills the words of `row` with `fill` and then, for a `masked` row, seals
/// its presence mask, the last word, which `fill` does not see. Every write
/// to the words of a row goes through here, so that no row is left with a
/// mask that does not match its words: a stale set bit over a zero word
/// would make native code follow a null slot.
#[inline(always)]
pub(crate) fn fill_sealed(row: &mut [usize], masked: bool, fill: impl FnOnce(&mut [usize])) {
    if masked {
        let words = row.len() - 1;
        fill(&mut row[..words]);
        seal_row(row);
    } else {
        fill(row);
    }
}

/// The words of a closure row of a trace that also reads table-valued cells
/// natively: the [`CLOSURE_ROW_WORDS`] words, then for each of the first
/// [`CLOSURE_ROW_CELLS`] cells one more word that holds the lane of the table
/// the cell held when the row was filled, plus one (zero when the cell holds
/// no table, or the run had no lane to give it).
pub const CLOSURE_ROW_TABLE_WORDS: usize = CLOSURE_ROW_WORDS + CLOSURE_ROW_CELLS;

/// Most lanes of a run's bank whose string length native code can read
/// without a call: the room a run has for strings it makes.
pub const MAX_TRACE_STRING_LANES: usize = 1024;

/// The index in the length table of the weight of the strings a run has made
/// since its last release: native code reads it to tell when a release is
/// due (see [`oximond::jit_ir::trace::TraceOp::ReleaseStrings`]).
pub const TRACE_PENDING_STRINGS: usize = MAX_TRACE_STRING_LANES;

/// A function native code calls for a host call, and the context it takes.
#[derive(Clone, Copy, Debug)]
pub struct HostEntry {
    /// The address of an `extern "C" fn(context: usize, args: *const
    /// TraceInputSlot, out: *mut TraceInputSlot, function: i64, argc: i64)`.
    pub function: usize,
    pub context: usize,
    /// The address of an `extern "C" fn(context: usize, words: *mut
    /// TraceInputSlot, binding: i64, key tag: i64, key bits: i64, value tag:
    /// i64, value bits: i64) -> i64` that makes a table set the way
    /// [`TraceTables::set`] does and returns 0, or -1 when it stored nothing.
    /// Zero for none.
    pub set: usize,
    /// The address of an `extern "C" fn(context: usize, binding: i64, tag:
    /// i64, bits: i64, slot: i64, replace: i64) -> i64` that stores the
    /// string or table the entry tag and bits name (a lane of the run's bank,
    /// or a root string) into the array slot at address `slot` of a table a
    /// [`BindingKind::Header`] binding made appendable: the table takes a
    /// counted reference to the string, or the handle of the table, with the
    /// collector's write barrier. `binding` is a binding of the register of
    /// the table. With `replace` zero the slot is spare capacity and gets the
    /// value without releasing anything; otherwise it holds a value, which is
    /// released. Returns 0, or -1 when it stored nothing. Zero for none.
    pub store: usize,
}

/// What the VM tells a run about the tables it meets.
///
/// A trace that loads a table out of an array, or reads fields of a table it
/// does not bind at entry, finds the table through a lane: the index of the
/// table in the run's root bank, the same id the exit restores from. Native
/// code asks [`Self::lane`] for the lane of a loaded table, and indexes a row
/// of field slot addresses by lane to read and overwrite a field in place.
///
/// The slot addresses a row holds follow the contract of [`TraceBinding`]:
/// they stay valid for the run because it changes table storage only by
/// overwriting a payload with a value of the same type, and nothing in it
/// runs Lua code or collects garbage. A trace that reads rows makes no table
/// set, the one operation that may resize a table.
pub trait TraceTables {
    /// The lane of the table `value` holds, adding it to the run's bank when
    /// it has none. `None` when `value` is no table, or the run has no lane
    /// left (see [`MAX_TRACE_TABLE_LANES`]): native code then leaves the
    /// trace at the load, and the interpreter does the access.
    fn lane(&mut self, value: &Value) -> Option<usize>;

    /// Builds the virtual tables the restore plan of exit `exit` builds, from
    /// the values native code spilled into the exit area at `area`, for a
    /// side that starts from the exit instead of the interpreter (see `build_entry_tables` in
    /// `mir/lower.rs`). The tables come out as the exit itself would make
    /// them, each with a lane of the run's bank, written to `lanes` in the
    /// plan's order (the order the snapshot's registers first name them).
    /// `elided` is the run's phantom debt so far (tables and bytes), which
    /// the host charges before it takes a table's own charge back, and
    /// zeroes. Returns how many lanes it wrote, or `None`, having made
    /// nothing a guest can see, when it cannot make them now: no plan, a
    /// collection owed, no lane or arena slot to spare, more tables than
    /// `lanes` holds. The guard then leaves through its exit.
    fn materialize_exit(
        &mut self,
        exit: usize,
        area: *const i64,
        elided: &mut (i64, i64),
        lanes: &mut [i64],
    ) -> Option<usize> {
        let _ = (exit, area, elided, lanes);
        None
    }

    /// Whether [`Self::admit_fresh`] serves this run.
    fn admits_fresh(&self) -> bool {
        false
    }

    /// [`Self::lane`] and [`Self::row`] in one for a run whose tables each
    /// get a lane of their own (a function trace's): pushes the table `value`
    /// holds to the bank and fills its row at once, in `rows` (`stride` words
    /// per lane, the last of them a presence mask if `masked`) and `filled`. `None` when the host has no such fast path for
    /// `value` (it is no table, or the run is not a function trace's), and
    /// `Some(-1)` when it has no lane left.
    fn admit_fresh(
        &mut self,
        value: &Value,
        rows: &mut [usize],
        stride: usize,
        masked: bool,
        filled: &mut [bool],
    ) -> Option<i64> {
        let _ = (value, rows, stride, masked, filled);
        None
    }

    /// The address of the run's index of table lanes, and the value the
    /// `ready` word of an entry holds while the entry names a table with a lane
    /// and a filled row. `None` for a host without one.
    ///
    /// The index is [`TRACE_LANE_INDEX_PLACES`] entries of 16 bytes: the
    /// table's identity (a handle's slot in the low half of the word, its
    /// generation in the high), its lane as a `u32`, and a `u32` ready word.
    /// A table sits at the place its identity hash names, or in one of the
    /// places after it. With `id` the identity, the hash is `h = id ^ (id >>
    /// 32)`, then `((h * 0x9E3779B97F4A7C15) >> 32) % places`, in wrapping
    /// 64-bit arithmetic. Native code looks at the first place only: an entry
    /// there that holds the identity with the ready word is the table's lane.
    /// Whatever else it finds, the dispatcher answers. The index stays where
    /// it is for the run, and no entry that is ready changes in it.
    fn lane_index(&mut self) -> Option<(usize, u32)> {
        None
    }

    /// The lane of the table `value` holds, with its row filled in `space`,
    /// as [`Self::lane`] and [`TraceTableSpace::ensure_row`] give it, in one
    /// call for a host that has a faster way to the same answer (`-1` when it
    /// has no lane, as for [`Self::admit_fresh`]). `None` leaves the value to
    /// those two. Native code asks each time it loads a table out of an array,
    /// so a loop over an array of records repeats the same few tables.
    fn admit_table(&mut self, value: &Value, space: &mut TraceTableSpace) -> Option<i64> {
        let _ = (value, space);
        None
    }

    /// Fills `row`, one slot per field of the trace's field list, for the
    /// table at `lane`: the address of the field's present value if it has
    /// the field's type, and zero otherwise (a missing key, another type, a
    /// table the interpreter may not write behind). Called once per lane and
    /// run, the first time native code could read the row.
    fn row(&mut self, lane: usize, row: &mut [usize]);

    /// Whether `row`, the words (not the mask) [`Self::row`] filled for
    /// `lane` before a call that ran arbitrary code, may stand after it:
    /// [`TraceTableSpace::refill`] then keeps it instead of filling it again.
    /// Every word it keeps must be one a fresh fill could give, or zero. No
    /// row stands by default.
    fn row_holds(&mut self, lane: usize, row: &[usize]) -> bool {
        let _ = (lane, row);
        false
    }

    /// Pushes the lane of every Lua closure the run's bank holds now, which a
    /// trace that reads and writes callee cells natively needs a row for
    /// before it starts.
    fn closure_lanes(&mut self, lanes: &mut Vec<usize>) {
        let _ = lanes;
    }

    /// Pushes the lanes whose rows the last [`Self::row`] call needs filled
    /// too (the tables a closure row names for its cells), and forgets them.
    fn row_followers(&mut self, lanes: &mut Vec<usize>) {
        let _ = lanes;
    }

    /// Sets `key` of the table bound as binding `binding` to `value` as a
    /// raw store ([`oximond::jit_ir::trace::TraceOp::ArraySet`]), the way the
    /// interpreter stores into a table with no `__newindex` for the key. Both
    /// arrive as scalar slots (see [`TraceInputSlot::scalar_value`]): a number
    /// key, and a nil, boolean or number value. Returns the table's array part
    /// as it stands after the store, which may have moved it. It must store
    /// nothing, and return `false`, when the store needs the interpreter: the
    /// key is absent and the table has a metatable, the key is NaN, the table
    /// is read-only. It runs no Lua code and collects no garbage. After a
    /// store it puts the table's array part, which the store may have moved,
    /// into `words` ([`TraceInputSlot::set_array`]).
    fn set(
        &mut self,
        binding: usize,
        key: TraceInputSlot,
        value: TraceInputSlot,
        words: &mut TraceInputSlot,
    ) -> bool {
        let _ = (binding, key, value, words);
        false
    }

    /// Looks up a string in the table bound as `binding`, a table with no
    /// metatable ([`oximond::jit_ir::trace::TraceOp::HashLoad`]). `key_tag` is
    /// [`TRACE_TAG_STRING`] with the string's lane in `key_bits`, or
    /// [`TRACE_TAG_BORROWED_STRING`] with the address of the slot holding it.
    /// Returns the value as a scalar slot: nil for a missing key or a stored
    /// nil, and [`TraceInputSlot::other`] for a value that is no scalar or
    /// when the lookup cannot be answered.
    ///
    /// The contract native code relies on: it is read-only, allocates
    /// nothing, collects nothing, runs no Lua code and never consults
    /// `__index`. It moves no table storage, so every address the run holds
    /// (including borrowed strings) stays valid.
    fn hash_get(&mut self, binding: usize, key_tag: i64, key_bits: i64) -> TraceInputSlot {
        let _ = (binding, key_tag, key_bits);
        TraceInputSlot::other()
    }

    /// Calls `function` on `args` ([`oximond::jit_ir::trace::TraceOp::HostCall`])
    /// and writes the answer to `out` as an entry tag and bits. A string
    /// answer is a new lane of the run's root bank. Returns `false`, having
    /// changed nothing a guest can see, when the call needs the interpreter:
    /// an error, a metamethod, an argument the call cannot take, or a run
    /// out of room for strings. It runs no Lua code and collects no garbage.
    ///
    /// A string answer also writes its byte count to `lengths[lane]`.
    fn host_call(
        &mut self,
        function: HostFunction,
        args: &[TraceInputSlot],
        out: &mut TraceInputSlot,
        lengths: &mut [u32],
    ) -> bool {
        let _ = (function, args, out, lengths);
        false
    }

    /// The function and context native code calls directly for a host call,
    /// when the implementation has one (`extern "C" fn(context, args, out,
    /// function, argc)`, see [`HostEntry`]). Without one the calls go through
    /// [`Self::host_call`].
    fn host_entry(&mut self) -> Option<HostEntry> {
        None
    }

    /// The address of the first value of the run's bank, which must not move
    /// until the run ends (the implementation reserves the room the run can
    /// add lanes in). Native code reads and overwrites scalar lanes in place.
    /// Zero (the default) means the host gives none, and code that wants it
    /// calls the helper instead.
    fn bank_address(&mut self) -> usize {
        0
    }

    /// The address of the trace's root strings and how many there are,
    /// which native code reads a string lane of the roots from (see
    /// [`TraceControl::roots`]). They do not move during a run. Zero (the
    /// default) means the host gives none, and code that wants them calls
    /// the helper instead.
    fn roots_address(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Writes the byte count of every string lane the bank holds at the start
    /// of a run to `lengths`, which native code reads for the length of a
    /// string it holds by lane.
    fn string_lengths(&mut self, lengths: &mut [u32]) {
        let _ = lengths;
    }

    /// Hands the implementation the run's flags of which lanes have a row
    /// (one per lane of [`MAX_TRACE_TABLE_LANES`]), for a run that makes
    /// tables. When it drops a table it made from a lane it clears the
    /// lane's flag, so that the next table the lane holds gets a row of its
    /// own instead of the stale one. The flags live as long as the run.
    fn row_flags(&mut self, filled: &mut [bool]) {
        let _ = filled;
    }

    /// Hands the implementation the run's row storage, for a run whose code
    /// calls the host and may have to make every row again afterwards.
    fn row_space(&mut self, space: &mut TraceTableSpace) {
        let _ = space;
    }

    /// Whether bindings `a` and `b` resolved the same table, so a store
    /// through one moves the array of the other.
    fn same_table(&mut self, a: usize, b: usize) -> bool {
        a == b
    }

    /// After [`Self::set`] stored through `binding`: rewrites the word of
    /// every length binding ([`oximond::jit_ir::trace::BindingKind::Length`])
    /// of the same table in `words`, the run's binding slots, since the store
    /// may have changed the table's border.
    fn rewrite_lengths(&mut self, binding: usize, words: &mut [TraceInputSlot]) {
        let _ = (binding, words);
    }
}

/// Row storage for the tables of one run, reused from run to run.
#[derive(Debug, Default)]
pub struct TraceTableSpace {
    rows: Vec<usize>,
    /// Lanes being listed, kept between runs.
    pub(crate) scratch: Vec<usize>,
    pub(crate) lengths: Vec<u32>,
    pub(crate) filled: Vec<bool>,
    fields: usize,
    /// Whether the last word of each row is its presence mask.
    pub(crate) masked: bool,
    /// The lanes a run may have.
    pub(crate) lanes: usize,
    /// Lanes being filled after a row, kept between runs.
    followers: Vec<usize>,
    /// The lanes [`TableDispatch::admit`] gave closures this run, direct
    /// mapped by the closure's identity: the identity and the lane, or
    /// [`NO_MEMO_LANE`] for an empty place. Empty until a run admits a
    /// closure.
    closure_memo: Vec<(u64, u32)>,
    /// Whether a place of `closure_memo` is taken.
    closure_memo_used: bool,
}

/// The places of [`TraceTableSpace::closure_memo`], a power of two.
const CLOSURE_MEMO_PLACES: usize = 64;
/// The lane of an empty place of [`TraceTableSpace::closure_memo`]: a lane
/// no row has, of either kind.
const NO_MEMO_LANE: u32 = u32::MAX;
/// The bit of a place's lane that says the place holds a coroutine.
const MEMO_THREAD: u32 = 1 << 31;

/// The place of [`TraceTableSpace::closure_memo`] for an identity. The two
/// kinds take different places for one identity, so a loop over closures
/// and coroutines of the same slots keeps both.
#[inline]
fn memo_place(identity: u64, thread: bool) -> usize {
    #[allow(clippy::cast_possible_truncation, reason = "masked to a place")]
    let place = identity as usize ^ (usize::from(thread) * (CLOSURE_MEMO_PLACES / 2));
    place & (CLOSURE_MEMO_PLACES - 1)
}

impl TraceTableSpace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Readies the space for a run of a trace whose rows have `fields` words:
    /// no lane has a row yet. The rows are not cleared: a row is only read
    /// after it is filled.
    pub(crate) fn begin(&mut self, fields: usize, lanes: usize, masked: bool) {
        self.fields = fields;
        self.masked = masked;
        self.lanes = lanes;
        // The rows are not cleared, so the room a run needs only grows.
        if self.rows.len() < lanes * fields {
            self.rows.resize(lanes * fields, 0);
        }
        self.filled.clear();
        self.filled.resize(lanes, false);
        // A lane belongs to its run.
        if self.closure_memo_used {
            self.closure_memo.fill((0, NO_MEMO_LANE));
            self.closure_memo_used = false;
        }
        if self.lengths.len() <= MAX_TRACE_STRING_LANES {
            self.lengths.resize(MAX_TRACE_STRING_LANES + 1, 0);
        }
    }

    /// Claims the row of `lane` for the host to fill with `fill`, which gets
    /// the row's words (not its mask word): returns whether the lane was claimed, that is, was inside the
    /// space and had no row yet. The host fills every word, as
    /// [`TraceTables::row`] does, and the presence mask is sealed here, so a
    /// filled row cannot go without one.
    #[inline(always)]
    pub(crate) fn claim_row(&mut self, lane: usize, fill: impl FnOnce(&mut [usize])) -> bool {
        if lane >= self.lanes || self.filled[lane] {
            return false;
        }
        self.filled[lane] = true;
        fill_sealed(
            &mut self.rows[lane * self.fields..(lane + 1) * self.fields],
            self.masked,
            fill,
        );
        true
    }

    /// The lanes a run may have.
    #[inline]
    pub(crate) const fn lanes(&self) -> usize {
        self.lanes
    }

    /// Fills the row of `lane` through `tables` unless it is filled already.
    /// A lane past the space has no row, and native code never indexes it.
    pub(crate) fn ensure_row(&mut self, lane: usize, tables: &mut dyn TraceTables) {
        if lane >= self.lanes || self.filled[lane] {
            return;
        }
        self.filled[lane] = true;
        let row = &mut self.rows[lane * self.fields..(lane + 1) * self.fields];
        fill_sealed(row, self.masked, |row| tables.row(lane, row));
        let mut followers = std::mem::take(&mut self.followers);
        followers.clear();
        tables.row_followers(&mut followers);
        for &follower in &followers {
            self.ensure_row(follower, tables);
        }
        self.followers = followers;
    }

    /// Fills every row of the first `lanes` lanes that is filled again
    /// through `tables`, for a run whose tables something else may have
    /// changed. Lanes the run's bank does not reach have no row.
    pub(crate) fn refill(&mut self, lanes: usize, tables: &mut dyn TraceTables) {
        let words = self.fields - usize::from(self.masked && self.fields != 0);
        for lane in 0..lanes.min(self.lanes) {
            if self.filled[lane] {
                // A row that still says what it said keeps its words, and
                // so its mask.
                let start = lane * self.fields;
                if tables.row_holds(lane, &self.rows[start..start + words]) {
                    continue;
                }
                self.filled[lane] = false;
                self.ensure_row(lane, tables);
            }
        }
    }

    /// Checks that the mask of every filled row is the one [`seal_row`] makes
    /// of its words. Debug builds run it when a run ends.
    #[cfg(debug_assertions)]
    pub(crate) fn assert_sealed(&self) {
        if self.fields == 0 || !self.masked {
            return;
        }
        for lane in (0..self.lanes).filter(|lane| self.filled[*lane]) {
            let row = &self.rows[lane * self.fields..(lane + 1) * self.fields];
            let mut sealed = row.to_vec();
            seal_row(&mut sealed);
            assert_eq!(
                row.last(),
                sealed.last(),
                "the presence mask of lane {lane} is stale"
            );
        }
    }

    /// Zeroes the row of `lane`, which the run does not fill yet: a row of
    /// zeros makes every access native code tries through it leave through
    /// its guard, and [`Self::ensure_row`] fills it when the lane is asked for.
    pub(crate) fn clear_row(&mut self, lane: usize) {
        if lane < self.lanes && !self.filled[lane] {
            self.rows[lane * self.fields..(lane + 1) * self.fields].fill(0);
        }
    }

    /// The lane the run gave the closure with `identity` (or, with
    /// `thread`, the coroutine), if its row is still filled. A closure and
    /// a coroutine may share an identity, so a place says which it holds in
    /// the top bit of its lane ([`MEMO_THREAD`]).
    #[inline]
    fn memo_closure(&self, identity: u64, thread: bool) -> Option<usize> {
        let &(known, lane) = self.closure_memo.get(memo_place(identity, thread))?;
        let held = (lane & !MEMO_THREAD) as usize;
        (known == identity
            && (lane & MEMO_THREAD != 0) == thread
            && self.filled.get(held).copied().unwrap_or(false))
        .then_some(held)
    }

    /// Remembers that the closure (or, with `thread`, the coroutine) with
    /// `identity` has `lane`, whose row is filled.
    fn remember_closure(&mut self, identity: u64, thread: bool, lane: usize) {
        let Ok(lane) = u32::try_from(lane) else {
            return;
        };
        if lane >= MEMO_THREAD {
            return;
        }
        if self.closure_memo.is_empty() {
            self.closure_memo
                .resize(CLOSURE_MEMO_PLACES, (0, NO_MEMO_LANE));
        }
        let kind = if thread { MEMO_THREAD } else { 0 };
        self.closure_memo[memo_place(identity, thread)] = (identity, lane | kind);
        self.closure_memo_used = true;
    }

    pub(crate) fn lengths_address(&mut self) -> usize {
        self.lengths.as_mut_ptr() as usize
    }

    pub(crate) fn rows_address(&mut self) -> usize {
        self.rows.as_mut_ptr() as usize
    }
}

/// The state native code reaches through [`TraceControl::tables`].
pub(crate) struct TableDispatch<'a> {
    pub(crate) space: &'a mut TraceTableSpace,
    pub(crate) tables: &'a mut dyn TraceTables,
    /// Which bindings of the run resolve an array part, by position.
    pub(crate) arrays: &'a [bool],
    /// The number of bindings, which the run's binding slots hold.
    pub(crate) bindings: usize,
    /// The lookups native code left to [`TraceTables::hash_get`] this run.
    pub(crate) hash_calls: i64,
    /// Whether the host admits tables with [`TraceTables::admit_fresh`].
    pub(crate) fresh: bool,
    /// Whether the host keeps an index of lanes ([`TraceTables::lane_index`]),
    /// the runs for which [`TraceTables::admit_table`] can answer. A run that
    /// makes tables has none, and skips the call.
    pub(crate) indexed: bool,
}

impl TableDispatch<'_> {
    /// [`TraceTables::hash_get`].
    pub(crate) fn hash_get(
        &mut self,
        binding: usize,
        key_tag: i64,
        key_bits: i64,
    ) -> TraceInputSlot {
        self.hash_calls += 1;
        self.tables.hash_get(binding, key_tag, key_bits)
    }

    /// [`TraceTables::host_call`] for the words native code passed.
    pub(crate) fn host_call(
        &mut self,
        function: i64,
        args: &[TraceInputSlot],
        out: &mut TraceInputSlot,
    ) -> bool {
        match HostFunction::from_code(function) {
            Some(function) => self
                .tables
                .host_call(function, args, out, &mut self.space.lengths),
            None => false,
        }
    }

    /// Fills the row of the table at `lane` again, for a table whose
    /// metatable the run changed.
    pub(crate) fn refresh_row(&mut self, lane: usize) {
        if lane < self.space.lanes {
            self.space.filled[lane] = false;
            self.space.ensure_row(lane, &mut *self.tables);
        }
    }

    /// Fills the row of the table at `lane`, a lane the run's bank has.
    pub(crate) fn ensure_row(&mut self, lane: usize) {
        if lane < self.space.lanes {
            self.space.ensure_row(lane, &mut *self.tables);
        }
    }

    /// [`TraceTables::materialize_exit`] for the run of `control`: the lanes
    /// go to the control block's `helper_out`, each with its row made again
    /// (the lane held another table before, or none).
    pub(crate) fn materialize_exit(
        &mut self,
        control: &mut TraceControl,
        area: *const i64,
        exit: usize,
    ) -> bool {
        let mut elided = (control.elided_tables, control.elided_bytes);
        let mut lanes = [0_i64; MAX_HELPER_RESULTS];
        let made = self
            .tables
            .materialize_exit(exit, area, &mut elided, &mut lanes);
        (control.elided_tables, control.elided_bytes) = elided;
        let Some(made) = made else {
            return false;
        };
        for &lane in &lanes[..made.min(MAX_HELPER_RESULTS)] {
            if let Ok(lane) = usize::try_from(lane) {
                self.refresh_row(lane);
            }
        }
        control.helper_out = lanes;
        true
    }

    /// [`TraceTables::admit_fresh`] over the run's rows, kept out of line so
    /// that [`Self::admit`] stays small for the runs that have no use for it.
    #[inline(never)]
    fn admit_fresh(&mut self, value: &Value) -> Option<i64> {
        self.tables.admit_fresh(
            value,
            &mut self.space.rows,
            self.space.fields,
            self.space.masked,
            &mut self.space.filled,
        )
    }

    /// [`TraceTables::admit_table`] over the run's space, kept out of line for
    /// the same reason as [`Self::admit_fresh`].
    #[inline(never)]
    fn admit_indexed(&mut self, value: &Value) -> Option<i64> {
        self.tables.admit_table(value, self.space)
    }

    /// The lane of a closure the run has met and whose row is still filled,
    /// from the memo and nothing else. [`Self::admit`] gives a closure the
    /// same lane: it remembers a closure only on its last path, which it
    /// reaches with no index left to ask, and a host that admits fresh
    /// tables passes closures on, so the memo holds an identity only when
    /// the paths before it give this lane. A caller asks here first so that
    /// the answer costs it no saved register.
    #[inline]
    pub(crate) fn admit_met(&self, value: &Value) -> Option<i64> {
        let Value::Closure(closure) = value else {
            return None;
        };
        let lane = self.space.memo_closure(closure.native_identity(), false)?;
        Some(i64::try_from(lane).unwrap_or(-1))
    }

    /// The lane of the table in `value`, with its row filled, or -1.
    #[inline]
    pub(crate) fn admit(&mut self, value: &Value) -> i64 {
        if self.fresh
            && let Some(lane) = self.admit_fresh(value)
        {
            return lane;
        }
        if self.indexed {
            if let Some(lane) = self.admit_indexed(value) {
                return lane;
            }
            // The host did not answer: its index gave up (more tables than it
            // holds), or the run makes tables, which the index leaves out.
            // The rest of the run goes the way a host without an index does.
            self.indexed = false;
        }
        // A loop that calls whichever closure an array holds asks for the
        // same few closures over and over. A closure keeps its lane for the
        // run (the bank roots it until the run ends), so the lane it got
        // answers again while its row is filled.
        // A loop that resumes whichever coroutine an array holds asks the
        // same way, and a coroutine keeps its lane for the run too.
        let closure = match value {
            Value::Closure(closure) => Some((closure.native_identity(), false)),
            Value::Thread(thread) => Some((thread.native_identity(), true)),
            _ => None,
        };
        if let Some((identity, thread)) = closure
            && let Some(lane) = self.space.memo_closure(identity, thread)
        {
            return i64::try_from(lane).unwrap_or(-1);
        }
        let Some(lane) = self.tables.lane(value) else {
            return -1;
        };
        if lane >= self.space.lanes {
            return -1;
        }
        self.space.ensure_row(lane, &mut *self.tables);
        if let Some((identity, thread)) = closure
            && self.space.filled[lane]
        {
            self.space.remember_closure(identity, thread, lane);
        }
        i64::try_from(lane).unwrap_or(-1)
    }

    /// Stores `value` at `key` of the table bound as `binding` and, when it
    /// did, rewrites `words` (one slot per binding) for every array binding:
    /// the store may have grown any of them, since two bindings may name one
    /// table.
    pub(crate) fn set(
        &mut self,
        binding: usize,
        key: TraceInputSlot,
        value: TraceInputSlot,
        words: &mut [TraceInputSlot],
    ) -> bool {
        if !self.arrays.get(binding).copied().unwrap_or(false)
            || !self.tables.set(binding, key, value, &mut words[binding])
        {
            return false;
        }
        // Word by word: the set wrote each with its own store.
        let (tag, bits) = (words[binding].tag, words[binding].bits);
        for (index, slot) in words.iter_mut().enumerate() {
            if index != binding && self.arrays[index] && self.tables.same_table(binding, index) {
                slot.tag = tag;
                slot.bits = bits;
            }
        }
        // A length binding is a binding of its own.
        if words.len() > 1 {
            self.tables.rewrite_lengths(binding, words);
        }
        true
    }
}

/// What kind of exit a descriptor stands for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceExitClass {
    /// A guard failed before any effect committed. The snapshot says where
    /// the interpreter resumes.
    GuardFail,
    /// A guard failed after an effect committed. The snapshot resumes after
    /// it, so the effect never runs twice.
    CommittedEffect,
    /// The loop ran to its natural end: the snapshot resumes at the loop's
    /// successor with the loop variables updated.
    LoopComplete,
    /// The fuel ran out between iterations. The snapshot publishes the whole
    /// state and resumes at the loop header. In a function trace: on entry to
    /// an activation, whose state is its parameters at pc 0.
    Poll,
    /// A function trace's activation whose self call's callee left the trace:
    /// the snapshot (the call's `Before` one) is this activation's state
    /// stopped at the call, which resumes after it once the callee returns.
    /// Native code never returns this index itself: it marks the exit of
    /// each activation around the one that left (see
    /// [`oximond::jit::CompiledTrace::activation_stride`]).
    Unwind,
}

/// What a failed guard checked, for statistics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceGuardClass {
    /// A value's dynamic type.
    Type,
    /// A branch direction.
    Branch,
    /// A call site's identity check: the live callee is not the closure the
    /// trace inlined.
    Callee,
    /// A table whose metatable is not the class the trace pinned
    /// ([`oximond::jit_ir::trace::TraceOp::MetaGuard`]): the table has no
    /// metatable, another one, or none its row can name.
    Class,
    /// A table access the trace cannot do natively: the key is outside the
    /// array, not an integral number, or the slot holds nil or another type,
    /// or a table has no such field, or the run has no lane for a table.
    Array,
    /// A self call of a function trace at the deepest activation a run may
    /// reach: the interpreter makes the call.
    Depth,
    /// A library call the VM could not answer inside the run: the call is
    /// one the interpreter has to make (an error, a metamethod, a pattern,
    /// another kind of result, no room left for strings).
    Host,
}

/// Where one spilled value lives in the exit area.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitSlot {
    pub value: ValueId,
    pub ty: TraceType,
    /// Index into the exit area. Unused for nil.
    pub slot: u32,
}

/// One way out of a native trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExitDescriptor {
    pub class: TraceExitClass,
    /// The trace of the tree this exit belongs to: 0 for the root, then the
    /// side traces in the order of [`oximond::jit_ir::trace::TraceIr::sides`].
    /// [`Self::snapshot`] indexes that member's snapshots
    /// ([`oximond::jit::CompiledTrace::lowered_member`]).
    pub member: usize,
    /// The origin of the guard this exit belongs to
    /// ([`oximond::jit_ir::trace::GuardInfo::origin`]), for guard exits. A side
    /// trace names the guard it continues by it.
    pub origin: Option<u32>,
    /// The snapshot this exit restores.
    pub snapshot: SnapshotId,
    /// The guard kind, for guard exits.
    pub guard: Option<TraceGuardClass>,
    /// The spilled values, sorted by value id.
    pub slots: Vec<ExitSlot>,
    /// The counted guest instructions the iteration ran before this exit,
    /// from [`oximond::jit_ir::trace::Snapshot::counted`]. Zero in a trace
    /// that is not counted.
    pub counted: u32,
}

impl ExitDescriptor {
    /// The scalar `value` held when the trace left, read from `area`.
    #[must_use]
    pub fn scalar(&self, area: &[i64], value: ValueId) -> Option<TraceScalar> {
        let index = self
            .slots
            .binary_search_by_key(&value, |slot| slot.value)
            .ok()?;
        let slot = self.slots[index];
        if slot.ty == TraceType::Nil {
            return Some(TraceScalar::Nil);
        }
        let bits = *area.get(slot.slot as usize)?;
        Some(match slot.ty {
            TraceType::Float => {
                TraceScalar::Float(f64::from_bits(u64::from_ne_bytes(bits.to_ne_bytes())))
            }
            TraceType::Integer => TraceScalar::Integer(bits),
            TraceType::Int64 => TraceScalar::Int64(bits),
            TraceType::UInt64 => TraceScalar::UInt64(bits.cast_unsigned()),
            TraceType::Boolean => TraceScalar::Boolean(bits != 0),
            TraceType::Table => TraceScalar::Table(bits.cast_unsigned()),
            TraceType::Closure => TraceScalar::Closure(bits.cast_unsigned()),
            TraceType::Thread => TraceScalar::Thread(bits.cast_unsigned()),
            TraceType::Userdata => TraceScalar::Userdata(bits.cast_unsigned()),
            TraceType::String => TraceScalar::String(bits.cast_unsigned()),
            TraceType::StringRef => TraceScalar::StringRef(bits.cast_unsigned()),
            TraceType::Nil | TraceType::Any => return None,
        })
    }

    /// Slots of the exit area this exit writes.
    #[must_use]
    pub fn area_len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.ty != TraceType::Nil)
            .map(|slot| slot.slot as usize + 1)
            .max()
            .unwrap_or(0)
    }
}

/// How a native run ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceOutcome {
    /// Entry input `input` did not have its required type. No trace
    /// operation ran.
    EntryMismatch { input: usize },
    /// The trace stores a string or a table into an array, and the host gave
    /// the run no store entry to call. No trace operation ran.
    StoreEntryMissing,
    /// The trace left through exit `index` of [`oximond::jit::CompiledTrace::exits`].
    Exit { index: usize },
    /// A function trace's activation returned (see
    /// [`oximond::jit_ir::trace::TraceTerminator::Return`]). Slot 0 of the exit
    /// area holds the value, as an exit slot of the function's result type
    /// holds it, when the function returns one.
    Returned,
}
