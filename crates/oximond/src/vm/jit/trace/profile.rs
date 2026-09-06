//! Portable recorded IR, with callable references resolved in the receiving VM.

use super::policy::{FUNCTION_ENTRY_PC, SiteState};
use super::recorder::HookPoint;
use super::{TracePolicy, TraceStatistics};
use crate::bytecode::Chunk;
use crate::jit_ir::ChunkDigest;
use crate::jit_ir::trace::{
    HostFunction, NativeBinary, NativeUnary, TraceIr, TraceOp, TraceScalar, pinned_builtin_kind,
    pinned_native_identity, pinned_native_index,
};
use crate::runtime::{ClosureHandle, NativeFunctionHandle, Runtime, Value};
use crate::vm::Vm;
use crate::vm::stack::FrameRef;
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::sync::Arc;

/// The start of every profile file: the name, then the format version. The
/// version changes whenever the encoding of the IR does (a removed or
/// renumbered host function or helper), so an older file is rejected.
const MAGIC: &[u8] = b"OXITRACE\x09";
const MAGIC_NAME: &[u8] = b"OXITRACE";
const MAX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TRACES: usize = 1_024;
const MAX_REFERENCES: usize = 4096;

fn invalid(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Root {
    Register(usize),
    Upvalue(usize),
    Environment,
    Closure,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Key {
    String(Vec<u8>),
    Integer(i64),
    Number(u64),
    Boolean(bool),
}

impl Key {
    fn of(value: &Value) -> Option<Self> {
        match value {
            Value::String(s) => Some(Self::String(s.as_bytes().to_vec())),
            Value::Integer(n) => Some(Self::Integer(*n)),
            Value::Number(n) => Some(Self::Number(n.to_bits())),
            Value::Boolean(b) => Some(Self::Boolean(*b)),
            _ => None,
        }
    }
    fn value(&self) -> Value {
        match self {
            Self::String(s) => Value::from(s.as_slice()),
            Self::Integer(n) => Value::Integer(*n),
            Self::Number(n) => Value::Number(f64::from_bits(*n)),
            Self::Boolean(b) => Value::Boolean(*b),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Step {
    Field(Key),
    Cell(usize),
    Environment,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Path {
    root: Root,
    steps: Vec<Step>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Callable {
    Closure { path: Path, digest: [u8; 32] },
    Native { path: Path, kind: Option<u32> },
    Builtin(u32),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Calls {
    pinned: Vec<Callable>,
    /// Native numeric operations, in preheader/body traversal order.
    numeric: Vec<Callable>,
    sides: Vec<Calls>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RecordedTrace {
    digest: [u8; 32],
    header: usize,
    ir: TraceIr,
    calls: Calls,
}

#[derive(Serialize, Deserialize)]
struct File {
    compatibility: String,
    traces: Vec<RecordedTrace>,
}

#[derive(Debug, Default)]
pub(super) struct ProfileState {
    pub(super) recording: bool,
    /// Runtime references are copied while the interpreter still holds them.
    paths: HashMap<([u8; 32], usize), HashMap<u64, Callable>>,
    recorded_chunks: HashMap<usize, (std::sync::Weak<Chunk>, [u8; 32])>,
    recorded: Vec<RecordedTrace>,
    loaded: Vec<RecordedTrace>,
    attempted: HashSet<(usize, usize)>,
    digests: HashMap<usize, (std::sync::Weak<Chunk>, [u8; 32])>,
}

struct Context<'a> {
    runtime: &'a Runtime,
    registers: &'a [Value],
    upvalues: &'a [crate::runtime::CellHandle],
    environment: crate::runtime::TableHandle,
    closure: Option<ClosureHandle>,
}

impl Context<'_> {
    fn resolve(&self, path: &Path) -> Option<Value> {
        let mut value = match path.root {
            Root::Register(r) => self.registers.get(r)?.clone(),
            Root::Upvalue(i) => self.runtime.cell_get(*self.upvalues.get(i)?).ok()?,
            Root::Environment => Value::Table(self.environment),
            Root::Closure => Value::Closure(self.closure?),
        };
        for step in &path.steps {
            value = match (step, value) {
                (Step::Field(key), Value::Table(table)) => {
                    self.runtime.table_get_raw_hit(table, &key.value())?.clone()
                }
                (Step::Cell(i), Value::Closure(closure)) => {
                    let parts = self.runtime.closure_call_parts(closure)?;
                    self.runtime.cell_get(*parts.cells.get(*i)?).ok()?
                }
                (Step::Environment, Value::Closure(closure)) => {
                    Value::Table(self.runtime.closure_environment(closure).ok()?)
                }
                _ => return None,
            };
        }
        Some(value)
    }

    fn callable_roots(&self, ir: &TraceIr) -> VecDeque<(Value, Path)> {
        let mut queue = VecDeque::new();
        queue.push_back((
            Value::Table(self.environment),
            Path {
                root: Root::Environment,
                steps: vec![],
            },
        ));
        if let Some(closure) = self.closure {
            queue.push_back((
                Value::Closure(closure),
                Path {
                    root: Root::Closure,
                    steps: vec![],
                },
            ));
        }
        for (i, cell) in self.upvalues.iter().enumerate() {
            if let Ok(value) = self.runtime.cell_get(*cell) {
                queue.push_back((
                    value,
                    Path {
                        root: Root::Upvalue(i),
                        steps: vec![],
                    },
                ));
            }
        }
        for (i, value) in self.registers.iter().enumerate() {
            // Scratch registers written by the trace need not be initialized
            // on the first iteration. Only entry inputs are valid roots.
            if !ir
                .entry
                .inputs
                .iter()
                .any(|input| usize::from(input.register) == i)
            {
                continue;
            }
            queue.push_back((
                value.clone(),
                Path {
                    root: Root::Register(i),
                    steps: vec![],
                },
            ));
        }
        queue
    }

    /// Breadth-first raw traversal. No Lua callbacks or metamethods run here.
    fn callables(&self, ir: &TraceIr) -> HashMap<u64, Callable> {
        let mut queue = self.callable_roots(ir);
        let mut seen = HashSet::new();
        let mut found = HashMap::new();
        let mut visited = 0;
        while let Some((value, path)) = queue.pop_front() {
            visited += 1;
            if visited > MAX_REFERENCES {
                break;
            }
            match value {
                Value::NativeFunction(handle) => {
                    found
                        .entry(pinned_native_identity(handle.index()))
                        .or_insert(Callable::Native {
                            path,
                            kind: self
                                .runtime
                                .direct_native_kind(handle)
                                .map(|kind| kind as u32),
                        });
                }
                Value::Closure(handle) => {
                    if !seen.insert((0, handle.native_identity())) {
                        continue;
                    }
                    let Some(parts) = self.runtime.closure_call_parts(handle) else {
                        continue;
                    };
                    found.insert(
                        handle.native_identity(),
                        Callable::Closure {
                            path: path.clone(),
                            digest: *ChunkDigest::of(parts.chunk).as_bytes(),
                        },
                    );
                    if path.steps.len() >= 16 {
                        continue;
                    }
                    for (i, cell) in parts.cells.iter().enumerate() {
                        if let Ok(value) = self.runtime.cell_get(*cell) {
                            let mut next = path.clone();
                            next.steps.push(Step::Cell(i));
                            queue.push_back((value, next));
                        }
                    }
                    let mut next = path;
                    next.steps.push(Step::Environment);
                    queue.push_back((Value::Table(parts.environment), next));
                }
                Value::Table(handle) => {
                    if !seen.insert((1, handle.native_identity())) || path.steps.len() >= 16 {
                        continue;
                    }
                    let mut previous = Value::Nil;
                    while queue.len() + visited < MAX_REFERENCES {
                        let Ok((key, value)) = self.runtime.table_next(handle, &previous) else {
                            break;
                        };
                        if key == Value::Nil {
                            break;
                        }
                        if let Some(key) = Key::of(&key) {
                            let mut next = path.clone();
                            next.steps.push(Step::Field(key));
                            queue.push_back((value, next));
                        }
                        previous = key;
                    }
                }
                _ => {}
            }
        }
        found
    }
}

fn describe(ir: &TraceIr, paths: &HashMap<u64, Callable>, runtime: &Runtime) -> Option<Calls> {
    let pinned = ir
        .pinned_closures
        .iter()
        .map(|id| {
            pinned_builtin_kind(*id)
                .map(Callable::Builtin)
                .or_else(|| paths.get(id).cloned())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut numeric = Vec::new();
    for instruction in ir.preheader.iter().chain(&ir.body) {
        let id = match instruction.op {
            TraceOp::NativeUnary { function, .. } => paths.keys().find(|id| {
                pinned_native_index(**id)
                    .and_then(|i| runtime.native_numeric_unary(NativeFunctionHandle::from_index(i)))
                    .is_some_and(|f| f as usize == function.address())
            }),
            TraceOp::NativeBinary { function, .. } => paths.keys().find(|id| {
                pinned_native_index(**id)
                    .and_then(|i| {
                        runtime.native_numeric_binary(NativeFunctionHandle::from_index(i))
                    })
                    .is_some_and(|f| f as usize == function.address())
            }),
            TraceOp::HostCall {
                function: HostFunction::Native(index),
                ..
            } => paths
                .keys()
                .find(|id| pinned_native_index(**id) == Some(index)),
            TraceOp::Constant(TraceScalar::StringRef(_)) => return None,
            _ => continue,
        }?;
        numeric.push(paths.get(id)?.clone());
    }
    Some(Calls {
        pinned,
        numeric,
        sides: ir
            .sides
            .iter()
            .map(|s| describe(&s.ir, paths, runtime))
            .collect::<Option<_>>()?,
    })
}

fn callable_value(call: &Callable, context: &Context<'_>) -> Option<Value> {
    match call {
        Callable::Closure { path, digest } => {
            let value = context.resolve(path)?;
            let Value::Closure(handle) = value else {
                return None;
            };
            (ChunkDigest::of(context.runtime.closure_call_parts(handle)?.chunk).as_bytes()
                == digest)
                .then_some(value)
        }
        Callable::Native { path, kind } => match context.resolve(path)? {
            value @ Value::NativeFunction(handle)
                if context
                    .runtime
                    .direct_native_kind(handle)
                    .map(|kind| kind as u32)
                    == *kind =>
            {
                Some(value)
            }
            _ => None,
        },
        Callable::Builtin(_) => None,
    }
}

fn relocate(ir: &mut TraceIr, calls: &Calls, context: &Context<'_>) -> Option<()> {
    if calls.pinned.len() != ir.pinned_closures.len() || calls.sides.len() != ir.sides.len() {
        return None;
    }
    for (id, call) in ir.pinned_closures.iter_mut().zip(&calls.pinned) {
        *id = match call {
            Callable::Builtin(kind) => crate::jit_ir::trace::pinned_builtin_identity(*kind),
            _ => match callable_value(call, context)? {
                Value::Closure(handle) => handle.native_identity(),
                Value::NativeFunction(handle) => pinned_native_identity(handle.index()),
                _ => return None,
            },
        };
    }
    let mut numeric = calls.numeric.iter();
    for instruction in ir.preheader.iter_mut().chain(&mut ir.body) {
        match &mut instruction.op {
            TraceOp::NativeUnary { function, .. } => {
                let Value::NativeFunction(handle) = callable_value(numeric.next()?, context)?
                else {
                    return None;
                };
                *function = NativeUnary(context.runtime.native_numeric_unary(handle)?);
            }
            TraceOp::NativeBinary { function, .. } => {
                let Value::NativeFunction(handle) = callable_value(numeric.next()?, context)?
                else {
                    return None;
                };
                *function = NativeBinary(context.runtime.native_numeric_binary(handle)?);
            }
            TraceOp::HostCall {
                function: HostFunction::Native(index),
                ..
            } => {
                let Value::NativeFunction(handle) = callable_value(numeric.next()?, context)?
                else {
                    return None;
                };
                *index = handle.index();
            }
            TraceOp::Constant(TraceScalar::StringRef(_)) => return None,
            _ => {}
        }
    }
    if numeric.next().is_some() {
        return None;
    }
    for (side, calls) in ir.sides.iter_mut().zip(&calls.sides) {
        relocate(&mut side.ir, calls, context)?;
    }
    Some(())
}

impl TracePolicy {
    fn export_profile(&self, runtime: &Runtime) -> Vec<RecordedTrace> {
        let mut records = self.trace_profile.recorded.clone();
        for trace in self.cache.profile_entries() {
            let (address, header) = trace.profile_address();
            let Some((_, digest)) = self.trace_profile.recorded_chunks.get(&address) else {
                continue;
            };
            let digest = *digest;
            let Some(paths) = self.trace_profile.paths.get(&(digest, header)) else {
                continue;
            };
            let ir = trace.profile_ir();
            let Some(calls) = describe(ir, paths, runtime) else {
                continue;
            };
            let record = RecordedTrace {
                digest,
                header,
                ir: ir.clone(),
                calls,
            };
            if let Some(old) = records.iter_mut().find(|old| {
                old.digest == digest
                    && old.header == header
                    && old.ir.entry == ir.entry
                    && old.ir.pinned_closures == ir.pinned_closures
            }) {
                *old = record;
            } else if records.len() < MAX_TRACES {
                records.push(record);
            }
        }
        records
    }
    pub(super) fn capture_profile(
        &mut self,
        at: &HookPoint<'_>,
        chunk: &Arc<Chunk>,
        header: usize,
        ir: &TraceIr,
        closure: Option<ClosureHandle>,
        root: bool,
    ) {
        if !self.trace_profile.recording {
            return;
        }
        let Some(around) = at.surroundings else {
            return;
        };
        let key = (*ChunkDigest::of(chunk).as_bytes(), header);
        let address = Arc::as_ptr(chunk) as usize;
        if !self.trace_profile.recorded_chunks.contains_key(&address)
            && self.trace_profile.recorded_chunks.len() >= MAX_TRACES
        {
            return;
        }
        self.trace_profile
            .recorded_chunks
            .entry(address)
            .or_insert_with(|| (Arc::downgrade(chunk), key.0));
        if !self.trace_profile.paths.contains_key(&key)
            && self.trace_profile.paths.len() >= MAX_TRACES
        {
            return;
        }
        let context = Context {
            runtime: around.runtime,
            registers: at.registers,
            upvalues: around.upvalues,
            environment: around.environment,
            closure,
        };
        let paths = self.trace_profile.paths.entry(key).or_default();
        paths.extend(
            context
                .callables(ir)
                .into_iter()
                .filter(|(id, _)| ir.pinned_closures.contains(id)),
        );
        if root {
            self.remember_profile_tree(chunk, header, ir, around.runtime);
        }
    }

    pub(super) fn remember_profile_tree(
        &mut self,
        chunk: &Arc<Chunk>,
        header: usize,
        ir: &TraceIr,
        runtime: &Runtime,
    ) {
        if !self.trace_profile.recording {
            return;
        }
        let digest = *ChunkDigest::of(chunk).as_bytes();
        let Some(paths) = self.trace_profile.paths.get(&(digest, header)) else {
            return;
        };
        let Some(calls) = describe(ir, paths, runtime) else {
            return;
        };
        let record = RecordedTrace {
            digest,
            header,
            ir: ir.clone(),
            calls,
        };
        if let Some(old) = self.trace_profile.recorded.iter_mut().find(|old| {
            old.digest == digest
                && old.header == header
                && old.ir.entry == ir.entry
                && old.ir.pinned_closures == ir.pinned_closures
        }) {
            *old = record;
        } else if self.trace_profile.recorded.len() < MAX_TRACES {
            self.trace_profile.recorded.push(record);
        }
    }

    pub(crate) fn has_loaded_profile(&self) -> bool {
        !self.trace_profile.loaded.is_empty()
    }

    pub(crate) fn enter_profile(
        &mut self,
        stats: &mut TraceStatistics,
        frame: FrameRef<'_>,
        runtime: &Runtime,
        profile: crate::compat::CompatibilityProfile,
    ) -> Option<u32> {
        if !self.enabled || self.recorder.is_some() || !self.has_loaded_profile() {
            return None;
        }
        let chunk = frame.chunk.as_arc();
        let pc = frame.pc;
        let address = Arc::as_ptr(&chunk) as usize;
        if self.trace_profile.digests.len() >= 1024
            && !self.trace_profile.digests.contains_key(&address)
        {
            self.trace_profile.digests.retain(|address, (owner, _)| {
                let alive = owner.strong_count() != 0;
                if !alive {
                    self.trace_profile
                        .attempted
                        .retain(|(site, _)| site != address);
                }
                alive
            });
            if self.trace_profile.digests.len() >= 1024 {
                return None;
            }
        }
        // Weak ownership prevents address reuse from matching a previous chunk.
        let digest = self
            .trace_profile
            .digests
            .entry(address)
            .or_insert_with(|| (Arc::downgrade(&chunk), *ChunkDigest::of(&chunk).as_bytes()))
            .1;
        let candidates: Vec<_> = self
            .trace_profile
            .loaded
            .iter()
            .enumerate()
            .filter(|(version, record)| {
                record.digest == digest
                    && (record.header == pc || (record.header == FUNCTION_ENTRY_PC && pc == 0))
                    && !self.trace_profile.attempted.contains(&(address, *version))
            })
            .map(|(version, record)| (version, record.clone()))
            .collect();
        if candidates.is_empty() {
            return self.profile_index(&chunk, pc);
        }
        let context = Context {
            runtime,
            registers: frame.registers,
            upvalues: &frame.upvalues,
            environment: frame.environment,
            closure: frame.closure,
        };
        for (version, record) in candidates {
            if self.trace_profile.attempted.contains(&(address, version)) {
                continue;
            }
            let mut ir = record.ir;
            if relocate(&mut ir, &record.calls, &context).is_none() {
                continue;
            }
            if ir.verify().is_err() {
                continue;
            }
            super::native::install_helpers();
            let Ok(compiled) = crate::jit::compile_trace_with(
                &ir,
                self.tier,
                crate::jit::TraceCompileOptions {
                    opt_flags: self.opt_flags,
                    count_links: self.count_links,
                },
            ) else {
                continue;
            };
            self.trace_profile.attempted.insert((address, version));
            self.profile = Some(profile);
            self.sites.site_mut(&chunk, record.header)?.state = SiteState::Compiling;
            self.install_compiled(stats, &chunk, record.header, Arc::new(ir), compiled);
        }
        self.profile_index(&chunk, pc)
    }

    fn profile_index(&mut self, chunk: &Arc<Chunk>, pc: usize) -> Option<u32> {
        self.sites
            .existing_mut(chunk, pc)
            .and_then(|s| s.head)
            .or_else(|| {
                (pc == 0)
                    .then(|| {
                        self.sites
                            .existing_mut(chunk, FUNCTION_ENTRY_PC)
                            .and_then(|s| s.head)
                    })
                    .flatten()
            })
    }
}

impl Vm {
    /// Records portable trace IR for subsequent runs. Small one-shot loops are
    /// recorded too, since the receiving run has already paid for discovery.
    pub fn start_trace_profile_recording(&mut self) {
        self.jit.trace.trace_profile.recording = true;
        self.jit.trace.judge_one_shot_loops = false;
    }

    /// Encodes recorded traces, with a version and compatibility check.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoded profile exceeds the 16 MiB limit.
    pub fn trace_profile(&self) -> io::Result<Vec<u8>> {
        let file = File {
            compatibility: format!("{:?}", self.compatibility),
            traces: self.jit.trace.export_profile(&self.runtime),
        };
        let encoded = bincode::DefaultOptions::new()
            .with_limit(MAX_BYTES)
            .serialize(&file)
            .map_err(invalid)?;
        let mut bytes = MAGIC.to_vec();
        bytes.extend(encoded);
        Ok(bytes)
    }

    /// Loads a recorded profile. Matching traces compile synchronously at their
    /// first entry, before the interpreter executes that header. Entry guards
    /// and ordinary interpreter fallbacks remain in force.
    ///
    /// Profiles are build-specific, trusted test artifacts. Regenerate them
    /// after changing the compiler, trace IR, or native library installation.
    ///
    /// # Errors
    ///
    /// Rejects malformed, oversized, invalid, or incompatible profiles, and
    /// attempts to load a profile outside tracing execution.
    pub fn load_trace_profile(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.execution_mode != crate::compat::ExecutionMode::Tracing {
            return Err(invalid("trace profile loading requires tracing execution"));
        }
        if bytes.len() as u64 > MAX_BYTES || !bytes.starts_with(MAGIC) {
            if bytes.len() as u64 <= MAX_BYTES && bytes.starts_with(MAGIC_NAME) {
                return Err(invalid(format!(
                    "trace profile format version {} is not supported (this build reads version {}); regenerate the profile",
                    bytes.get(MAGIC_NAME.len()).copied().unwrap_or(0),
                    MAGIC[MAGIC_NAME.len()],
                )));
            }
            return Err(invalid("invalid trace profile header or size"));
        }
        let file: File = bincode::DefaultOptions::new()
            .with_limit(MAX_BYTES)
            .reject_trailing_bytes()
            .deserialize(&bytes[MAGIC.len()..])
            .map_err(invalid)?;
        if file.compatibility != format!("{:?}", self.compatibility)
            || file.traces.len() > MAX_TRACES
        {
            return Err(invalid(
                "trace profile compatibility or trace count mismatch",
            ));
        }
        for record in &file.traces {
            record.ir.verify().map_err(invalid)?;
        }
        self.jit.trace.trace_profile.loaded = file.traces;
        self.jit.trace.trace_profile.attempted.clear();
        self.jit.trace.trace_profile.digests.clear();
        Ok(())
    }
}
