//! The `Runtime` heap owner, its configuration, and core allocation.

use crate::runtime::arena::Arena;
use crate::runtime::closures::{Cell, ClosureBody};
use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::files::{FileBuffering, LinesFormats, RuntimeFile};
use crate::runtime::gc::{
    GC_MIN_NURSERY_BYTES, GcMode, GcReference, GcRootBuilder, GcRoots, IncrementalGc,
    ParallelMarking, RuntimeThread,
};
use crate::runtime::helpers::NonFiniteNumerals;
use crate::runtime::iterators::GmatchIterator;
use crate::runtime::metatable::SharedMetatable;
use crate::runtime::number_format::{NumberFormat, format_integer, small_integer_string};
use crate::runtime::table_storage::Table;
use crate::runtime::table_storage::key::{TableKey, integer_to_number};
use crate::runtime::userdata::{FinalizerObject, TableFinalizerState, Userdata};
use crate::runtime::value::{
    CellHandle, DirectNativeKind, FileHandle, NativeEntry, TableHandle, ThreadHandle,
    UserdataHandle, Value,
};
use crate::string::LuaString;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;

/// Host warning callback. `continued` marks all but the final message fragment.
pub type WarningHandler = dyn FnMut(&[u8], bool) -> RuntimeResult<()>;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use std::time::Instant;

/// The empty slot of `Runtime::direct_natives`. Handles are indices into the
/// native table, which never reaches `u32::MAX` entries.
pub(crate) const NO_DIRECT_NATIVE: u32 = u32::MAX;
/// A handle with no recorded kind in `Runtime::direct_native_kinds`.
pub(crate) const NO_DIRECT_NATIVE_KIND: u8 = u8::MAX;
const _: () = assert!(DirectNativeKind::COUNT < NO_DIRECT_NATIVE_KIND as usize);

/// What the `*_identity` table constructors return when no table could be
/// made. A real identity never has this value: its slot is below `u32::MAX`.
pub const NO_TABLE_IDENTITY: u64 = u64::MAX;

/// The trace compiler parameters of `jit.opt.start`, each as the script gave
/// it. `None` leaves a parameter as it is.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JitOptions {
    /// `maxrecord`: most guest instructions a recording may follow.
    pub max_record: Option<u32>,
    /// `maxside`: most side traces of one tree.
    pub max_side: Option<u32>,
    /// `maxsnap`: most snapshots (guards) of one trace.
    pub max_snap: Option<u32>,
    /// `maxtrace`: most compiled traces.
    pub max_trace: Option<u32>,
    /// `hotloop`: backward branches before a loop is recorded.
    pub hot_loop: Option<u32>,
    /// `maxmcode`: kilobytes of machine code.
    pub max_mcode: Option<u32>,
}

impl JitOptions {
    /// Whether no parameter is set.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.max_record.is_none()
            && self.max_side.is_none()
            && self.max_snap.is_none()
            && self.max_trace.is_none()
            && self.hot_loop.is_none()
            && self.max_mcode.is_none()
    }

    /// These options with every parameter `later` sets replaced.
    #[must_use]
    pub const fn overlaid_by(self, later: Self) -> Self {
        const fn pick(earlier: Option<u32>, later: Option<u32>) -> Option<u32> {
            if later.is_some() { later } else { earlier }
        }
        Self {
            max_record: pick(self.max_record, later.max_record),
            max_side: pick(self.max_side, later.max_side),
            max_snap: pick(self.max_snap, later.max_snap),
            max_trace: pick(self.max_trace, later.max_trace),
            hot_loop: pick(self.hot_loop, later.hot_loop),
            max_mcode: pick(self.max_mcode, later.max_mcode),
        }
    }
}

/// The VM can mutate runtime state only through these methods. Heap layouts
/// and table storage remain private to this crate.
#[allow(clippy::struct_excessive_bools)] // These switches are independent compatibility and GC settings.
/// A host's hook that sets the host roots of its threads (see
/// [`Runtime::set_host_roots_refresher`]): a function and the token it is
/// called with, which names the host's state.
#[derive(Clone, Copy, Debug)]
pub struct HostRootsRefresher {
    pub refresh: fn(&mut Runtime, usize),
    pub token: usize,
}

/// The type names [`Value::type_name`] spells, by [`type_name_index`].
const TYPE_NAMES: [&str; 9] = [
    "nil", "boolean", "number", "string", "table", "function", "userdata", "thread", "cdata",
];

/// The position of `value`'s type name in [`TYPE_NAMES`].
#[inline]
const fn type_name_index(value: &Value) -> usize {
    match value {
        Value::Nil => 0,
        Value::Boolean(_) => 1,
        Value::Integer(_) | Value::Number(_) => 2,
        Value::String(_) => 3,
        Value::Table(_) => 4,
        Value::Closure(_)
        | Value::NativeFunction(_)
        | Value::BuiltinFunction(_)
        | Value::GmatchIterator(_)
        | Value::CoroutineWrapper(_) => 5,
        Value::File(_) | Value::Userdata(_) => 6,
        Value::Thread(_) => 7,
        Value::Int64(_) | Value::UInt64(_) => 8,
    }
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "independent compatibility switches, set once and read one at a time"
)]
pub struct Runtime {
    pub(crate) tables: Arena<Table>,
    pub(crate) natives: Vec<NativeEntry>,
    /// The library-qualified names (`math.floor`, `select`) the installers
    /// give natives, by handle index, and the library tables they name.
    pub(crate) native_names: crate::runtime::host::NativeNames,
    pub(crate) library_names: Vec<(TableHandle, &'static str)>,
    /// Emptied result vectors that natives take for their next call.
    pub(crate) result_pool: Vec<Vec<Value>>,
    /// The shared strings of the small integers, so that the interpreter
    /// reaches one without the thread-local lookup of
    /// [`small_integer_string`]. Filled on first use: building all of them
    /// costs a short script's startup about 4%.
    small_integer_strings: std::cell::OnceCell<Box<[LuaString]>>,
    /// The strings `type` returns, by [`type_name_index`], shared so that a
    /// call allocates nothing.
    type_name_strings: [LuaString; TYPE_NAMES.len()],
    pub(crate) closures: Arena<ClosureBody>,
    pub(crate) cells: Arena<Cell>,
    /// How many times a cell's value or a closure's cell was replaced
    /// ([`Self::cell_writes`]).
    pub(crate) cell_writes: u64,
    /// How many times a closure's cell was replaced or a cell reopened
    /// ([`Self::cell_rebinds`]).
    pub(crate) cell_rebinds: u64,
    pub(crate) upvalue_identities: BTreeMap<CellHandle, UserdataHandle>,
    pub(crate) files: Arena<RuntimeFile>,
    pub(crate) userdatas: Arena<Userdata>,
    pub(crate) pending_finalizers: VecDeque<FinalizerObject>,
    pub(crate) next_userdata_order: u64,
    pub(crate) next_finalizer_order: u64,
    pub(crate) gmatch_iterators: Arena<GmatchIterator>,
    /// Indexed by [`SharedMetatable`]: every lookup of a primitive's
    /// metatable (`tostring`, string methods) reads one slot.
    pub(crate) shared_metatables: [Option<TableHandle>; SharedMetatable::COUNT],
    pub(crate) default_input: Option<FileHandle>,
    pub(crate) default_output: Option<FileHandle>,
    pub(crate) global_environment: TableHandle,
    pub(crate) registry: TableHandle,
    // Lua's package library gives `require` a private package-table
    // environment. The global `package` binding is just another mutable
    // global, so removing it must not detach `require` from its loaders.
    pub(crate) package_require_table: Option<TableHandle>,
    /// Indexed by [`DirectNativeKind`]: the raw handle index, or
    /// [`NO_DIRECT_NATIVE`] when none is recorded. Raw `u32`s take half the
    /// space of `Option<NativeFunctionHandle>` in every runtime.
    pub(crate) direct_natives: [u32; DirectNativeKind::COUNT],
    /// The reverse of `direct_natives`, indexed by raw handle index: the
    /// first [`DirectNativeKind`] (in declaration order) recorded for the
    /// handle, or [`NO_DIRECT_NATIVE_KIND`]. Every native call that is not
    /// a library callback asks, so the answer must not scan every kind.
    pub(crate) direct_native_kinds: Vec<u8>,
    /// The probes of the natives that have one, by raw handle index.
    pub(crate) native_probes: Vec<(u32, crate::runtime::value::NativeProbe)>,
    /// The probes that give every result of a native.
    pub(crate) native_results_probes: Vec<(u32, crate::runtime::value::NativeResultsProbe)>,
    /// The probes that give every result of a native without a `Vec`.
    pub(crate) native_fixed_results_probes:
        Vec<(u32, crate::runtime::value::NativeFixedResultsProbe)>,
    /// The pure two-number form of each `math.log`, by handle index.
    pub(crate) native_log_bases: Vec<(u32, crate::runtime::value::NativeNumericBinary)>,
    /// The installed `setmetatable` native, as a raw handle index, or
    /// [`NO_DIRECT_NATIVE`]. The fast loop runs its common case inline.
    pub(crate) setmetatable_native: u32,
    pub(crate) output: Vec<u8>,
    pub(crate) error_output: Vec<u8>,
    pub(crate) output_writer: Option<Box<dyn Write>>,
    pub(crate) error_output_writer: Option<Box<dyn Write>>,
    pub(crate) output_buffering: FileBuffering,
    pub(crate) error_output_buffering: FileBuffering,
    pub(crate) warnings_enabled: bool,
    pub(crate) warning_handler: Option<Box<WarningHandler>>,
    pub(crate) warning_buffer: Vec<u8>,
    /// The last warning piece asked for more, so the next one continues it.
    pub(crate) warning_continues: bool,
    pub(crate) error_nil_as_string: bool,
    pub(crate) error_number_positions: bool,
    pub(crate) tagged_integers: bool,
    pub(crate) nonfinite_numerals: NonFiniteNumerals,
    /// The rules of the installed `tonumber`, for a trace that converts a
    /// string without calling the native. `None` until a library installs it.
    pub(crate) tonumber_rules: Option<crate::stdlib::TonumberRules>,
    /// Whether the installed `string.unpack` clamps its start position to
    /// the string (Lua 5.4 and later), for a trace that unpacks without
    /// calling the native. `None` until the string library installs it.
    pub(crate) unpack_clamped: Option<bool>,
    pub(crate) typed_argument_errors: bool,
    pub(crate) lines_formats: LinesFormats,
    pub(crate) number_format: NumberFormat,
    pub(crate) legacy_pattern_errors: bool,
    pub(crate) string_init_clamps: bool,
    /// Whether this VM can enter native code and whether native entry is enabled.
    pub(crate) jit_available: bool,
    pub(crate) jit_enabled: bool,
    /// Whether native execution is held off while a trace's call waits for
    /// the coroutine it resumed (see [`Self::suspend_jit`]).
    pub(crate) jit_suspended: bool,
    /// `jit_available && jit_enabled && !jit_suspended`, kept in step by the
    /// setters so that the interpreter's per-backedge test is one load.
    pub(crate) jit_effective: bool,
    pub(crate) jit_flush_requested: bool,
    /// Trace compiler limits `jit.opt.start` set that the VM has not applied.
    pub(crate) jit_options_pending: Option<JitOptions>,
    pub(crate) gc_pause: f64,
    pub(crate) gc_step_multiplier: f64,
    pub(crate) gc_minor_multiplier: f64,
    pub(crate) gc_major_multiplier: f64,
    pub(crate) gc_major_minor: Option<f64>,
    pub(crate) gc_step_bytes: Option<usize>,
    pub(crate) gc_major_start_bytes: usize,
    pub(crate) gc_running: bool,
    pub(crate) gc_debt: usize,
    /// Bytes charged for tables whose allocation scalar replacement elided.
    ///
    /// Elided tables never enter an arena, but `gcinfo` and
    /// `collectgarbage("count")` must still observe them as heap use that a
    /// completed collection releases, exactly as Lua 5.1's interpreter does.
    pub(crate) elided_table_bytes: usize,
    pub(crate) allocation_serial: u64,
    /// Tables allocated in the arena since the runtime started. Read by
    /// `OXIMOND_COUNTERS`; never reset, never consulted by the collector.
    pub(crate) tables_created: u64,
    /// Elided tables charged, and their bytes, since the runtime started.
    /// Unlike `elided_table_bytes` a finished cycle does not release them.
    pub(crate) elided_tables_total: u64,
    /// Bytes of charges already released by a cycle. Live charges are still
    /// in `elided_table_bytes`, which keeps the charge path to one add.
    pub(crate) elided_bytes_total: u64,
    pub(crate) gc_threshold: usize,
    /// Monotonically increasing reachability generation for the collector.
    pub(crate) gc_epoch: u64,
    /// Userdata freed by collection so far; see [`Runtime::gc_freed_userdata`].
    pub(crate) gc_freed_userdata: u64,
    pub(crate) incremental_gc: Option<Box<IncrementalGc>>,
    /// The mode `collectgarbage` reports.
    pub(crate) gc_mode: GcMode,
    /// Whether automatic collection runs minor collections between majors.
    pub(crate) gc_generational: bool,
    /// Consecutive minor collections in which most young objects survived.
    pub(crate) gc_bad_minors: u8,
    /// Trial minor collections that failed during the current pause; each
    /// one doubles the spacing to the next, up to one nursery.
    pub(crate) gc_trial_fails: u8,
    /// Length of the young table list at the last trial of this pause.
    pub(crate) gc_trial_prev_young: usize,
    /// Whether the trial that precedes a major cycle of this pause has run.
    pub(crate) gc_final_trial_done: bool,
    /// Big trials run since the last major cycle, before the final one.
    pub(crate) gc_big_trials: u8,
    /// Major cycles left before the default policy tries minor collections
    /// again; see [`Runtime::gc_minors_active`].
    pub(crate) gc_minor_pause: u32,
    /// The pause the next run of unproductive minor collections earns.
    pub(crate) gc_minor_backoff: u32,
    /// How many times the normal pace the current major cycle runs; above
    /// one only for the cycle that ends a pause of minor collections.
    pub(crate) gc_catch_up: usize,
    /// The debt at which a paused default policy starts a major cycle.
    pub(crate) gc_pause_major_debt: usize,
    /// Set once the old generation outgrew its budget; the next automatic
    /// collection starts a major cycle instead of a minor one.
    pub(crate) gc_major_requested: bool,
    /// The last automatic step stopped at the atomic step for want of roots.
    pub(crate) gc_awaiting_roots: bool,
    /// Brings the host roots of the threads up to date before collector work
    /// reads them (see [`Self::set_host_roots_refresher`]).
    pub(crate) host_roots_refresher: Option<HostRootsRefresher>,
    /// Old objects that received a reference to a non-old object since the
    /// last minor collection. Each has its arena `touched` flag set.
    pub(crate) gc_remembered: Vec<GcReference>,
    /// Old objects the next minor collection must trace once more.
    pub(crate) gc_carry: Vec<GcReference>,
    /// Estimated bytes in the old generation.
    pub(crate) gc_old_bytes: usize,
    /// Old-generation bytes after the last major cycle.
    pub(crate) gc_major_base: usize,
    /// The table an incremental major cycle is tracing in chunks, and
    /// whether the mutator changed it since the trace started.
    pub(crate) gc_partial_table: Option<TableHandle>,
    pub(crate) gc_partial_dirty: bool,
    /// Large dead tables whose storage is released over later steps.
    pub(crate) gc_detached: VecDeque<Table>,
    /// Tracing on several threads in the stop-the-world phases; see
    /// `gc/parallel.rs`.
    pub(crate) gc_parallel: ParallelMarking,
    /// Objects with a registered finalizer, by registration order.
    pub(crate) finalizer_registry: BTreeMap<u64, FinalizerObject>,
    /// Tables registered for or awaiting finalization.
    pub(crate) table_finalizers: HashMap<TableHandle, TableFinalizerState>,
    /// Whether `setmetatable` registers tables whose metatable has `__gc`.
    pub(crate) table_finalizers_enabled: bool,
    /// Whether weak-key tables are ephemerons, as in Lua 5.2 and later: an
    /// entry keeps its value alive only while its key is reachable. Lua 5.1
    /// and LuaJIT mark the values of weak-key tables strongly.
    pub(crate) ephemerons_enabled: bool,
    pub(crate) random_state: u64,
    pub(crate) random_modern_state: [u64; 4],
    pub(crate) random_modern: bool,
    pub(crate) random_luajit_state: [u64; 4],
    pub(crate) random_luajit: bool,
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    pub(crate) started_at: Instant,
    pub(crate) threads: Arena<RuntimeThread>,
    /// Interned `__index` key so native metamethod walks avoid per-read
    /// string allocations.
    pub(crate) index_metamethod_key: LuaString,
    /// Interned `__len` key for native border reads.
    pub(crate) len_metamethod_key: LuaString,
    /// Interned `__mode` key for weak-table checks during tracing.
    pub(crate) mode_metamethod_key: LuaString,
    /// The `__newindex`, `__gc` and `__metatable` keys, for the same reason.
    pub(crate) newindex_metamethod_key: LuaString,
    pub(crate) gc_metamethod_key: LuaString,
    pub(crate) metatable_metamethod_key: LuaString,
    /// Generation of the inherited-read cache; see `oximond::runtime::inherit`.
    pub(crate) inherit_epoch: u64,
    pub(crate) inherit_cache: Vec<crate::runtime::inherit::InheritEntry>,
    /// Operator metamethod handlers by metatable and event; see
    /// `oximond::runtime::operator_cache`. Valid for `inherit_epoch`.
    pub(crate) operator_cache: Vec<crate::runtime::operator_cache::OperatorEntry>,
    /// The `__newindex` closure that the last declining
    /// [`Runtime::table_set_string_if_no_newindex_hinted`] found, for the
    /// caller to take with [`Runtime::take_newindex_hint`].
    pub(crate) newindex_hint: Option<crate::runtime::value::ClosureHandle>,
    /// Trims the thread's list of one-entry blocks after the tables above
    /// have dropped. Fields drop in order, so it stays last.
    _pool_trim: PoolTrim,
}

/// Trims the dropping thread's lists of recycled table blocks.
#[derive(Default)]
pub(crate) struct PoolTrim;

impl Drop for PoolTrim {
    fn drop(&mut self) {
        crate::runtime::table_storage::ordered_vec::trim_pool();
        crate::runtime::table_storage::array::trim_pool();
    }
}

impl Default for Runtime {
    fn default() -> Self {
        let mut tables = Arena::default();
        let global_environment = TableHandle(
            tables
                .insert(Table::default())
                .expect("a fresh table arena has a first slot"),
        );
        let environment = tables
            .get_mut(global_environment.0)
            .expect("the fresh global table is present");
        environment.entries.insert(
            TableKey::string(&LuaString::from("_G")),
            Value::Table(global_environment),
        );
        let registry = TableHandle(
            tables
                .insert(Table::default())
                .expect("a fresh table arena has a second slot"),
        );
        Self {
            tables,
            natives: Vec::new(),
            native_names: crate::runtime::host::NativeNames::default(),
            library_names: Vec::new(),
            result_pool: Vec::new(),
            small_integer_strings: std::cell::OnceCell::new(),
            type_name_strings: TYPE_NAMES.map(LuaString::from),
            closures: Arena::default(),
            cells: Arena::default(),
            cell_writes: 0,
            cell_rebinds: 0,
            upvalue_identities: BTreeMap::new(),
            files: Arena::default(),
            userdatas: Arena::default(),
            pending_finalizers: VecDeque::new(),
            next_userdata_order: 0,
            next_finalizer_order: 0,
            gmatch_iterators: Arena::default(),
            shared_metatables: [None; SharedMetatable::COUNT],
            default_input: None,
            default_output: None,
            global_environment,
            registry,
            package_require_table: None,
            direct_natives: [NO_DIRECT_NATIVE; DirectNativeKind::COUNT],
            direct_native_kinds: Vec::new(),
            native_probes: Vec::new(),
            native_results_probes: Vec::new(),
            native_fixed_results_probes: Vec::new(),
            native_log_bases: Vec::new(),
            setmetatable_native: NO_DIRECT_NATIVE,
            output: Vec::new(),
            error_output: Vec::new(),
            output_writer: None,
            error_output_writer: None,
            output_buffering: FileBuffering::Line(8_192),
            error_output_buffering: FileBuffering::Line(8_192),
            warnings_enabled: true,
            warning_handler: None,
            warning_buffer: Vec::new(),
            warning_continues: false,
            error_nil_as_string: false,
            error_number_positions: false,
            tagged_integers: false,
            nonfinite_numerals: NonFiniteNumerals::Rejected,
            tonumber_rules: None,
            unpack_clamped: None,
            typed_argument_errors: false,
            lines_formats: LinesFormats::Ignored,
            number_format: NumberFormat::Lua51,
            legacy_pattern_errors: true,
            string_init_clamps: true,
            jit_available: false,
            jit_enabled: false,
            jit_suspended: false,
            jit_effective: false,
            jit_flush_requested: false,
            jit_options_pending: None,
            gc_pause: 200.0,
            gc_step_multiplier: 200.0,
            gc_minor_multiplier: 25.0,
            gc_major_multiplier: 100.0,
            gc_major_minor: None,
            gc_step_bytes: None,
            gc_major_start_bytes: 0,
            gc_running: true,
            gc_debt: 0,
            elided_table_bytes: 0,
            allocation_serial: 0,
            tables_created: 0,
            elided_tables_total: 0,
            elided_bytes_total: 0,
            gc_threshold: GC_MIN_NURSERY_BYTES,
            gc_epoch: 1,
            gc_freed_userdata: 0,
            incremental_gc: None,
            gc_mode: GcMode::Incremental,
            gc_generational: true,
            gc_bad_minors: 0,
            gc_trial_fails: 0,
            gc_trial_prev_young: 0,
            gc_final_trial_done: false,
            gc_big_trials: 0,
            gc_minor_pause: 0,
            gc_minor_backoff: 1,
            gc_catch_up: 1,
            gc_pause_major_debt: 0,
            gc_major_requested: false,
            gc_awaiting_roots: false,
            host_roots_refresher: None,
            gc_remembered: Vec::new(),
            gc_carry: Vec::new(),
            gc_old_bytes: 0,
            gc_major_base: 0,
            gc_partial_table: None,
            gc_partial_dirty: false,
            gc_detached: VecDeque::new(),
            gc_parallel: ParallelMarking::from_env(),
            finalizer_registry: BTreeMap::new(),
            table_finalizers: HashMap::new(),
            table_finalizers_enabled: false,
            ephemerons_enabled: false,
            random_state: 0x9e37_79b9_7f4a_7c15,
            random_modern_state: [0, 0xff, 0, 0],
            random_modern: false,
            random_luajit_state: [
                0xa0d2_7757_0a34_5b8c,
                0x764a_296c_5d4a_a64f,
                0x5122_0704_070a_deaa,
                0x2a27_17b5_a7b7_b927,
            ],
            random_luajit: false,
            #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
            started_at: Instant::now(),
            threads: Arena::default(),
            index_metamethod_key: crate::string::shared_key(b"__index"),
            len_metamethod_key: crate::string::shared_key(b"__len"),
            mode_metamethod_key: crate::string::shared_key(b"__mode"),
            newindex_metamethod_key: crate::string::shared_key(b"__newindex"),
            gc_metamethod_key: crate::string::shared_key(b"__gc"),
            metatable_metamethod_key: crate::string::shared_key(b"__metatable"),
            inherit_epoch: 1,
            inherit_cache: Vec::new(),
            operator_cache: Vec::new(),
            newindex_hint: None,
            _pool_trim: PoolTrim,
        }
    }
}

impl Runtime {
    /// Reports whether native execution is currently enabled.
    // Deliberate name: the getter pairs with `set_jit_enabled` and answers
    // the same question after hold-off, which `jit_effective` tracks.
    #[allow(clippy::misnamed_getters)]
    #[must_use]
    pub const fn jit_enabled(&self) -> bool {
        self.jit_effective
    }

    /// Reports whether the program asked for native execution and the VM
    /// supports it, whether or not it is held off just now. This is what
    /// `jit.status` tells a script.
    #[must_use]
    pub const fn jit_requested(&self) -> bool {
        self.jit_available && self.jit_enabled
    }

    /// Enables or disables native execution when the VM supports it.
    /// Interpreter-only VMs remain disabled.
    pub const fn set_jit_enabled(&mut self, enabled: bool) {
        self.jit_enabled = self.jit_available && enabled;
        self.refresh_jit();
    }

    const fn refresh_jit(&mut self) {
        self.jit_effective = self.jit_available && self.jit_enabled && !self.jit_suspended;
    }

    /// Holds native execution off, for the coroutine a native trace resumes:
    /// the code the coroutine runs is entered, recorded and compiled by
    /// nothing until [`Self::restore_jit`], since the trace that is running
    /// owns the engine's state. A script's `jit.on` and `jit.off` in the
    /// meantime still take effect. Returns whether it was already held off.
    pub const fn suspend_jit(&mut self) -> bool {
        let was = self.jit_suspended;
        self.jit_suspended = true;
        self.refresh_jit();
        was
    }

    /// Ends a [`Self::suspend_jit`], given what it returned.
    pub const fn restore_jit(&mut self, was: bool) {
        self.jit_suspended = was;
        self.refresh_jit();
    }

    /// Sets the native execution capability when a VM takes ownership of this runtime.
    /// The VM selects this from its execution mode.
    pub fn configure_jit(&mut self, available: bool) {
        self.jit_available = available;
        self.jit_enabled = available;
        self.refresh_jit();
    }

    /// Requests that the owning VM discard compiled code at its next safe point.
    /// A request made inside native code is applied after that code returns.
    pub const fn request_jit_flush(&mut self) {
        self.jit_flush_requested = true;
    }

    /// Reports whether the owning VM has a pending cache flush.
    #[must_use]
    pub const fn jit_flush_requested(&self) -> bool {
        self.jit_flush_requested
    }

    /// Consumes a pending cache flush at a VM safe point.
    pub const fn take_jit_flush_request(&mut self) -> bool {
        let requested = self.jit_flush_requested;
        self.jit_flush_requested = false;
        requested
    }

    /// Asks the owning VM to apply `options` (the parameters of
    /// `jit.opt.start`) at its next safe point. Compiled code is discarded
    /// then, since traces recorded under other limits are not comparable.
    /// Parameters the VM already has change nothing.
    pub fn request_jit_options(&mut self, options: JitOptions) {
        if !self.jit_available {
            return;
        }
        let merged = match self.jit_options_pending.take() {
            Some(pending) => pending.overlaid_by(options),
            None => options,
        };
        self.jit_options_pending = Some(merged);
        self.jit_flush_requested = true;
    }

    /// Takes the parameters `jit.opt.start` left for the VM.
    pub const fn take_jit_options(&mut self) -> Option<JitOptions> {
        self.jit_options_pending.take()
    }

    /// Selects whether integral library results use Lua's tagged integer representation.
    pub const fn set_tagged_integers(&mut self, enabled: bool) {
        self.tagged_integers = enabled;
    }

    /// Reports whether this runtime represents integral numbers as a distinct
    /// tagged value.
    #[must_use]
    pub const fn tagged_integers(&self) -> bool {
        self.tagged_integers
    }

    /// Selects which spellings of infinity and NaN library functions
    /// accept as numbers.
    pub const fn set_nonfinite_numerals(&mut self, numerals: NonFiniteNumerals) {
        self.nonfinite_numerals = numerals;
    }

    /// Records whether the `string.unpack` native the string library
    /// installs clamps its start position.
    pub(crate) const fn set_unpack_clamped(&mut self, clamped: bool) {
        self.unpack_clamped = Some(clamped);
    }

    /// Whether the installed `string.unpack` clamps its start position, if
    /// a library installed one.
    #[must_use]
    pub(crate) const fn unpack_clamped(&self) -> Option<bool> {
        self.unpack_clamped
    }

    /// Records the rules of the `tonumber` native the base library installs.
    pub(crate) const fn set_tonumber_rules(&mut self, rules: crate::stdlib::TonumberRules) {
        self.tonumber_rules = Some(rules);
    }

    /// The rules of the installed `tonumber` native, if a library set them.
    #[must_use]
    pub(crate) const fn tonumber_rules(&self) -> Option<crate::stdlib::TonumberRules> {
        self.tonumber_rules
    }

    /// Reports which spellings of infinity and NaN library functions
    /// accept as numbers.
    #[must_use]
    pub const fn nonfinite_numerals(&self) -> NonFiniteNumerals {
        self.nonfinite_numerals
    }

    /// Selects whether argument checks that Lua 5.4 reports through
    /// `luaL_typeerror`, such as `setmetatable`'s "nil or table", name the
    /// type they found.
    pub const fn set_typed_argument_errors(&mut self, enabled: bool) {
        self.typed_argument_errors = enabled;
    }

    /// Reports whether those argument checks name the type they found.
    #[must_use]
    pub const fn typed_argument_errors(&self) -> bool {
        self.typed_argument_errors
    }

    /// An argument error that Lua 5.4 reports as "`expected` expected, got
    /// ..." and older Lua as "`expected` expected".
    #[must_use]
    pub fn expected_argument_error(
        &self,
        position: usize,
        expected: &'static str,
        argument: Option<&Value>,
    ) -> RuntimeError {
        if self.typed_argument_errors {
            RuntimeError::bad_argument(position, expected, argument)
        } else {
            RuntimeError::argument_error(position, format!("{expected} expected"))
        }
    }

    /// Selects whether `io.lines` and `file:lines` forward read formats.
    pub const fn set_lines_formats(&mut self, formats: LinesFormats) {
        self.lines_formats = formats;
    }

    /// Reports whether `io.lines` and `file:lines` forward read formats.
    #[must_use]
    pub const fn lines_formats(&self) -> LinesFormats {
        self.lines_formats
    }

    /// Selects the wording of pattern errors: Lua 5.1 and LuaJIT when `legacy`.
    pub const fn set_legacy_pattern_errors(&mut self, legacy: bool) {
        self.legacy_pattern_errors = legacy;
    }

    /// Whether pattern errors read as in Lua 5.1 and LuaJIT.
    #[must_use]
    pub const fn legacy_pattern_errors(&self) -> bool {
        self.legacy_pattern_errors
    }

    /// Selects whether `string.find` and `string.match` start a search past
    /// the end of the subject at its end (Lua 5.1, LuaJIT) or fail it (5.2 and
    /// later). This follows the profile, not the number format: 5.2 spells
    /// floats like 5.1 but fails the search like 5.3.
    pub const fn set_string_init_clamps(&mut self, clamps: bool) {
        self.string_init_clamps = clamps;
    }

    /// Whether a search start past the end of the subject is clamped to its end.
    #[must_use]
    pub const fn string_init_clamps(&self) -> bool {
        self.string_init_clamps
    }

    /// Selects how `tostring`, concatenation, and string coercions spell floats.
    pub const fn set_number_format(&mut self, format: NumberFormat) {
        self.number_format = format;
    }

    /// The shared string of a small non-negative integer, as
    /// [`small_integer_string`] returns it, without a thread-local lookup.
    #[must_use]
    #[inline(always)]
    pub fn small_integer_text(&self, value: u64) -> Option<LuaString> {
        let strings = self
            .small_integer_strings
            .get_or_init(|| (0..).map_while(small_integer_string).collect());
        usize::try_from(value)
            .ok()
            .and_then(|index| strings.get(index))
            .cloned()
    }

    /// The shared string of `value`'s type name, as [`Value::type_name`]
    /// spells it.
    #[must_use]
    #[inline]
    pub fn type_name_string(&self, value: &Value) -> LuaString {
        self.type_name_strings[type_name_index(value)].clone()
    }

    /// [`Self::type_name_string`] by reference, for a caller that stores it
    /// itself.
    #[must_use]
    #[inline(always)]
    pub(crate) fn type_name_ref(&self, value: &Value) -> &LuaString {
        &self.type_name_strings[type_name_index(value)]
    }

    /// Reports how this runtime spells floats.
    #[must_use]
    pub const fn number_format(&self) -> NumberFormat {
        self.number_format
    }

    /// Creates a profile-appropriate integral numeric value.
    #[must_use]
    pub fn integer_value(&self, value: i64) -> Value {
        if self.tagged_integers {
            Value::Integer(value)
        } else {
            Value::Number(integer_to_number(value))
        }
    }

    /// Returns an integral float the way Lua 5.3+ `math.floor` and friends
    /// do: as an integer when it fits, otherwise as the float itself.
    /// Profiles without tagged integers keep the float.
    #[must_use]
    pub fn integral_float_value(&self, value: f64) -> Value {
        if self.tagged_integers
            && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
        {
            #[allow(clippy::cast_possible_truncation)] // Integral and within i64 bounds above.
            return Value::Integer(value as i64);
        }
        Value::Number(value)
    }

    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates a table with a stable handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its table-handle space.
    pub fn create_table(&mut self) -> RuntimeResult<TableHandle> {
        self.create_table_sized(0)
    }

    /// Makes room for `additional` tables, so that the next ones do not move
    /// the tables that exist. A trace that makes tables asks before it binds
    /// the address of any table.
    pub fn reserve_table_slots(&mut self, additional: usize) {
        self.tables.reserve_slots(additional);
    }

    /// Makes table creation fail, as an exhausted handle space does, once
    /// `additional` tables more than the free slots hold exist. For tests of
    /// the paths that meet that failure.
    #[doc(hidden)]
    pub fn limit_table_growth(&mut self, additional: u32) {
        self.tables.limit_growth(additional);
    }

    /// Whether a table can be made without moving the others
    /// ([`Self::reserve_table_slots`]).
    #[must_use]
    pub fn table_slot_is_spare(&self) -> bool {
        self.tables.has_spare_slot()
    }

    /// [`Self::table_slot_is_spare`] for `count` tables.
    #[must_use]
    pub fn table_slots_are_spare(&self, count: usize) -> bool {
        self.tables.has_spare_slots(count)
    }

    /// Allocates a table whose array part has room for `array` values, so a
    /// constructor that stores that many never regrows it.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its table-handle space.
    pub fn create_table_sized(&mut self, array: usize) -> RuntimeResult<TableHandle> {
        Self::table_from_identity(self.create_table_sized_identity(array))
    }

    /// [`Self::create_table_sized`] that returns the table's
    /// [`TableHandle::native_identity`], or [`NO_TABLE_IDENTITY`] when the
    /// handle space is exhausted. A result of one machine word comes back in
    /// a register: a `RuntimeResult` comes back in memory, written with
    /// narrow stores and read with a wide load that cannot forward from them.
    // Out of line, so that the shaped constructor beside it stays free of
    // the array part's buffer, whose pieces the optimizer would otherwise
    // merge with the empty array of a record and copy through the stack.
    #[inline(never)]
    pub fn create_table_sized_identity(&mut self, array: usize) -> u64 {
        let key = self.tables.insert_init(|slot| {
            Table::init_parts(
                slot,
                || crate::runtime::table_storage::array::ArrayPart::with_capacity(array),
                crate::runtime::table_storage::entries::TableEntries::default,
            );
        });
        self.finish_table_creation(key)
    }

    /// Allocates a table whose array part is `count` nil slots that count
    /// as present (at least one), and gives the address of the first with
    /// it: [`Self::create_table_sized`] and [`Self::table_prefill_array`] in
    /// one step, with one lookup of the table and no growth of its array.
    /// `None` when the handle space is exhausted.
    #[inline(never)]
    pub fn create_table_prefilled(&mut self, count: usize) -> Option<(TableHandle, usize)> {
        self.create_table_prefilled_inline(count)
    }

    /// [`Self::create_table_prefilled`] in line, for the trace helper that
    /// makes a table per call: the handle and the address come back in
    /// registers, and the helper saves its registers once for both.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn create_table_prefilled_inline(
        &mut self,
        count: usize,
    ) -> Option<(TableHandle, usize)> {
        debug_assert!(count != 0);
        let mut slots = 0;
        let key = self
            .tables
            .insert_init(|slot| slots = Table::init_prefilled(slot, count))?;
        self.tables_created += 1;
        self.gc_charge(crate::runtime::table_storage::TABLE_CHARGED_BYTES);
        Some((TableHandle(key), slots))
    }

    /// [`Self::create_table_with_shape`] that also gives the address of the
    /// table's first value slot ([`Self::table_shaped_slots`]), from the
    /// one look at the table that makes it. `None` when the handle space is
    /// exhausted.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) fn create_table_with_shape_slots(
        &mut self,
        shape: &std::sync::Arc<crate::runtime::TableShape>,
    ) -> Option<(TableHandle, usize)> {
        let mut slots = 0;
        let key = self
            .tables
            .insert_init(|slot| slots = Table::init_shaped(slot, shape))?;
        self.tables_created += 1;
        self.gc_charge(crate::runtime::table_storage::TABLE_CHARGED_BYTES);
        Some((TableHandle(key), slots))
    }

    #[inline]
    fn finish_table_creation(&mut self, key: Option<crate::runtime::arena::ArenaKey>) -> u64 {
        match key {
            Some(key) => {
                self.tables_created += 1;
                self.gc_charge(crate::runtime::table_storage::TABLE_CHARGED_BYTES);
                TableHandle(key).native_identity()
            }
            None => NO_TABLE_IDENTITY,
        }
    }

    fn table_from_identity(identity: u64) -> RuntimeResult<TableHandle> {
        if identity == NO_TABLE_IDENTITY {
            return Err(RuntimeError::message("table handle space exhausted"));
        }
        Ok(TableHandle::from_native_identity(identity))
    }

    /// Allocates a table whose string keys are those of `shape`, all absent.
    /// Charges what [`Self::create_table`] does: the value slots are charged
    /// as their keys are first stored, as an ordered table's entries are.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its table-handle space.
    // Out of line: inlined into the VM's dispatch loops it grows them, and
    // slows instructions that have nothing to do with constructors.
    #[inline(never)]
    pub fn create_table_with_shape(
        &mut self,
        shape: &std::sync::Arc<crate::runtime::TableShape>,
    ) -> RuntimeResult<TableHandle> {
        Self::table_from_identity(self.create_table_with_shape_identity(shape))
    }

    /// [`Self::create_table_with_shape`] that returns an identity, as
    /// [`Self::create_table_sized_identity`] does.
    #[inline]
    pub fn create_table_with_shape_identity(
        &mut self,
        shape: &std::sync::Arc<crate::runtime::TableShape>,
    ) -> u64 {
        let key = self.tables.insert_init(|slot| {
            Table::init_parts(
                slot,
                crate::runtime::table_storage::array::ArrayPart::new,
                || crate::runtime::table_storage::entries::TableEntries::with_shape_ref(shape),
            );
        });
        self.finish_table_creation(key)
    }

    /// Allocates a table wrapped in a [`Value`].
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its table-handle space.
    pub fn create_table_value(&mut self) -> RuntimeResult<Value> {
        self.create_table().map(Value::Table)
    }

    /// Allocates a VM-owned Lua thread identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has exhausted its thread-handle space.
    pub fn create_thread(&mut self) -> RuntimeResult<ThreadHandle> {
        let handle = self
            .threads
            .insert(RuntimeThread {
                roots: GcRootBuilder::new().finish(),
                host_roots: GcRootBuilder::new().finish(),
            })
            .map(ThreadHandle)
            .ok_or_else(|| RuntimeError::message("thread handle space exhausted"))?;
        self.gc_charge(std::mem::size_of::<RuntimeThread>());
        Ok(handle)
    }

    /// Replaces the heap roots retained by a new or suspended thread.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` does not name a live thread.
    pub fn set_thread_roots(&mut self, handle: ThreadHandle, roots: GcRoots) -> RuntimeResult<()> {
        self.threads
            .get(handle.0)
            .ok_or_else(|| RuntimeError::message("invalid thread"))?;
        let container = Some(GcReference::Thread(handle));
        for reference in roots
            .references
            .iter()
            .copied()
            .chain(roots.cells.iter().map(|cell| GcReference::Cell(*cell)))
            .chain(
                roots
                    .closures
                    .iter()
                    .map(|closure| GcReference::Closure(*closure)),
            )
        {
            self.gc_barrier(container, reference);
        }
        self.threads
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::message("invalid thread"))?
            .roots = roots;
        Ok(())
    }

    /// Installs `refresher`, which the collector calls with its token before
    /// every collection, step and full cycle, and which brings the host
    /// roots of the threads up to date (with [`Self::set_thread_host_roots`]).
    /// A host whose stacks change at every call (a C API, where every
    /// `lua_call` pushes and pops values) then builds its roots once per
    /// piece of collector work instead of once per call. Only the collector
    /// reads host roots, and the refresh runs before it does: what a step
    /// sees is what an eager update at the host's last call would have left,
    /// with the write barriers of that update. `None` removes it.
    pub fn set_host_roots_refresher(&mut self, refresher: Option<HostRootsRefresher>) {
        self.host_roots_refresher = refresher;
    }

    /// Runs the host roots refresher, if one is installed.
    pub(crate) fn refresh_host_roots(&mut self) {
        if let Some(refresher) = self.host_roots_refresher {
            (refresher.refresh)(self, refresher.token);
        }
    }

    /// Replaces host stack roots independently of the thread's VM frames.
    /// These roots are released when the thread is collected.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` does not name a live thread.
    pub fn set_thread_host_roots(
        &mut self,
        handle: ThreadHandle,
        roots: GcRoots,
    ) -> RuntimeResult<()> {
        self.threads
            .get(handle.0)
            .ok_or_else(|| RuntimeError::message("invalid thread"))?;
        let container = Some(GcReference::Thread(handle));
        for reference in roots
            .references
            .iter()
            .copied()
            .chain(roots.cells.iter().map(|cell| GcReference::Cell(*cell)))
            .chain(
                roots
                    .closures
                    .iter()
                    .map(|closure| GcReference::Closure(*closure)),
            )
        {
            self.gc_barrier(container, reference);
        }
        self.threads
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::message("invalid thread"))?
            .host_roots = roots;
        Ok(())
    }

    #[must_use]
    pub fn thread_exists(&self, handle: ThreadHandle) -> bool {
        self.threads.get(handle.0).is_some()
    }

    /// Formats a value using Lua's primitive `tostring` representation.
    ///
    /// This deliberately does not consult `__tostring`; the VM does that
    /// before falling back to this formatter.
    ///
    /// # Errors
    ///
    /// Returns an error when the result buffer cannot reserve its required
    /// space.
    ///
    /// # Panics
    ///
    /// Panics if `CInt64::from_value` cannot read a `Value::Int64` or
    /// `Value::UInt64` back; the match routes only those two variants to it.
    pub fn format_value(&self, value: &Value) -> RuntimeResult<LuaString> {
        let bytes = match value {
            Value::Nil => b"nil".to_vec(),
            Value::Boolean(value) => value.to_string().into_bytes(),
            Value::Integer(value) => {
                return Ok(u64::try_from(*value)
                    .ok()
                    .and_then(small_integer_string)
                    .unwrap_or_else(|| LuaString::from(format_integer(*value).as_bytes())));
            }
            Value::Number(value) => return Ok(self.number_format.format_string(*value)),
            Value::String(value) => return Ok(value.clone()),
            Value::Table(value) => format!("table: {}", value.index()).into_bytes(),
            Value::File(value) => format!("file: {}", value.index()).into_bytes(),
            Value::Userdata(value) => format!("userdata: {}", value.index()).into_bytes(),
            Value::GmatchIterator(value) => {
                format!("function: gmatch {}", value.index()).into_bytes()
            }
            Value::Thread(value) => format!("thread: {}", value.index()).into_bytes(),
            Value::CoroutineWrapper(value) => {
                format!("function: coroutine {}", value.index()).into_bytes()
            }
            Value::Closure(value) => format!("function: closure {}", value.index()).into_bytes(),
            Value::NativeFunction(value) => {
                format!("function: native {}", value.index()).into_bytes()
            }
            Value::BuiltinFunction(value) => format!("function: {value:?}").into_bytes(),
            Value::Int64(_) | Value::UInt64(_) => crate::runtime::cint64::CInt64::from_value(value)
                .expect("a 64-bit integer value")
                .to_text()
                .into_bytes(),
        };
        Ok(LuaString::from(bytes))
    }
}
