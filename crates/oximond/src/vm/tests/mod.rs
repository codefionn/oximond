// helpers shared by the split test modules
use super::*;
use super::{
    ActiveCallSite, DebugHookEvent, Frame, LoadCompletion, LoadPipeline, LoadReaderContinuation,
    ReturnTarget, TraceFrame, Traceback, Vm, VmError,
};
use crate::bytecode::load::{ChunkLoad, ChunkLoader, LoadInput, LoadPoll};
use crate::bytecode::{
    CallKind, CallSite, Chunk, Constant, DebugMetadata, DiagnosticSite, Instruction, OpCode,
    UpvalueDescriptor, UpvalueSource, ValidationError, binary,
};
use crate::compat::CompatibilityProfile;
use crate::compat::ExecutionMode;
use crate::jit::{JitBroker, OptimizationTier};
use crate::runtime::{
    BuiltinFunction, GcRootBuilder, Runtime, RuntimeError, RuntimeResult, TableHandle, Value,
};
use crate::stdlib::{install_base, install_base_with_profile};
use crate::string::LuaString;
use crate::vm::debug_info::{debug_short_source, format_traceback};
use crate::vm::jit::inline_table::InlineTablePlan;
use crate::vm::jit::state::JIT_QUEUE_CAPACITY;
use crate::vm::registers::numeric_for_instruction;
use std::collections::BTreeSet;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static VM_CALL_BRIDGE_EFFECTS: AtomicUsize = AtomicUsize::new(0);

static VM_CALL_BRIDGE_TEST_LOCK: Mutex<()> = Mutex::new(());

static CLOSE_HANDLER_ORDER: Mutex<Vec<f64>> = Mutex::new(Vec::new());

fn record_vm_call_bridge_effect(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    let [Value::Number(value)] = arguments else {
        return Err(RuntimeError::message(
            "bridge recorder received the wrong arguments",
        ));
    };
    VM_CALL_BRIDGE_EFFECTS.fetch_add(1, Ordering::SeqCst);
    Ok(vec![Value::Number(*value)])
}

/// Allocates garbage until an automatic collection is due, whatever the
/// collector's current threshold. Keeps the collector's running state.
/// Makes `cell` the open upvalue of register `register` of frame `frame`,
/// wherever it stood before: the cell then reads and writes that register.
#[allow(unsafe_code)]
fn open_cell_at_register(
    vm: &mut Vm,
    frames: &mut crate::vm::stack::CallStack,
    frame: usize,
    register: u8,
    cell: crate::runtime::CellHandle,
) {
    let slot = frames[frame].base + u32::from(register);
    let anchor = frames.anchor();
    // SAFETY: the slot is a register of a live frame of `frames`.
    unsafe { vm.runtime_mut().reopen_cell(cell, anchor, slot) };
    frames.push_open(slot, cell);
}

fn make_gc_due(runtime: &mut Runtime) {
    let was_running = runtime.gc_is_running();
    runtime.restart_gc();
    for _ in 0..1_000_000 {
        if runtime.gc_needs_collection() {
            break;
        }
        runtime.create_table().unwrap();
    }
    assert!(runtime.gc_needs_collection());
    if !was_running {
        runtime.stop_gc();
    }
}

/// A tracing VM with a synchronous broker, so trace compiles install at
/// predictable points.
fn synchronous_tracing_vm() -> Vm {
    let mut vm = Vm::with_profile_and_execution_mode(
        CompatibilityProfile::default(),
        ExecutionMode::Tracing,
    );
    vm.jit.broker = Some(JitBroker::synchronous(JIT_QUEUE_CAPACITY));
    vm
}

fn inline_increment_callee() -> Chunk {
    Chunk::with_parts(
        2,
        1,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 1, 0, 1),
            Instruction::return_(1),
        ],
        Vec::new(),
        Vec::new(),
    )
}

fn one_argument_call_one_parent(destination: u8) -> Chunk {
    Chunk::with_parts(
        3,
        2,
        Vec::new(),
        vec![
            Instruction::call_one(destination, 0, 1),
            Instruction::return_(destination),
        ],
        Vec::new(),
        Vec::new(),
    )
}

/// A deliberately non-inlineable Lua callee. Its numeric inner loop can
/// compile on its own, while the recorder after that loop proves that a
/// VM-call side exit never executes the Lua call twice.
fn vm_call_bridge_callee(delta: f64) -> Chunk {
    Chunk::with_parts(
        8,
        1,
        vec![
            Constant::Number(1.0),
            Constant::Number(delta),
            Constant::String("bridge_record".into()),
        ],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::load_constant(3, 0),
            Instruction::move_(5, 0),
            Instruction::numeric_for_prep(1, 2),
            Instruction::binary(OpCode::Add, 5, 5, 4),
            Instruction::aj(OpCode::Jump, 0).unwrap(),
            Instruction::numeric_for_loop(1, -3),
            Instruction::get_global(6, 2),
            Instruction::move_(7, 5),
            Instruction::call(6, 1, 1),
            Instruction::return_(6),
        ],
        Vec::new(),
        Vec::new(),
    )
}

/// Splits an outer numeric loop around an ordinary non-inlineable
/// `CallOne`. PC 5 is native work before the call and PC 7 is native work
/// after it, so a VM-call continuation must make both fragments hot.
fn vm_call_bridge_caller(limit: f64) -> Chunk {
    Chunk::with_parts(
        8,
        1,
        vec![
            Constant::Number(1.0),
            Constant::Number(limit),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::load_constant(3, 0),
            Instruction::load_constant(7, 2),
            Instruction::numeric_for_prep(1, 4),
            Instruction::move_(5, 4),
            Instruction::call_one(6, 0, 5),
            Instruction::binary(OpCode::Add, 7, 7, 6),
            Instruction::aj(OpCode::Jump, 0).unwrap(),
            Instruction::numeric_for_loop(1, -5),
            Instruction::return_(7),
        ],
        Vec::new(),
        Vec::new(),
    )
}

fn vm_call_bridge_expected(limit: u32, delta: u32) -> f64 {
    let limit = f64::from(limit);
    let delta = f64::from(delta);
    limit * (limit + 1.0) / 2.0 + limit * delta * (delta + 1.0) / 2.0
}

fn captured_counter_call_one_callee() -> Chunk {
    Chunk::with_parts(
        2,
        1,
        Vec::new(),
        vec![
            Instruction::get_upvalue(1, 0),
            Instruction::binary(OpCode::Add, 1, 1, 0),
            Instruction::set_upvalue(1, 0),
            Instruction::get_upvalue(0, 0),
            Instruction::return_(0),
        ],
        Vec::new(),
        vec![UpvalueDescriptor {
            name: "value".into(),
            source: UpvalueSource::Detached,
        }],
    )
}

fn numeric_return_chunk() -> Chunk {
    Chunk::new(
        3,
        vec![Constant::Number(2.0), Constant::Number(3.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
    )
}

struct FunctionLoader(fn(&LuaString, &LuaString) -> Result<Chunk, String>);

impl ChunkLoader for FunctionLoader {
    fn begin(&self, source_name: LuaString) -> Box<dyn ChunkLoad> {
        Box::new(FunctionLoad {
            compile: self.0,
            source_name,
            bytes: Vec::new(),
        })
    }
}

struct FunctionLoad {
    compile: fn(&LuaString, &LuaString) -> Result<Chunk, String>,
    source_name: LuaString,
    bytes: Vec<u8>,
}

impl ChunkLoad for FunctionLoad {
    fn resume(mut self: Box<Self>, input: LoadInput) -> LoadPoll {
        match input {
            LoadInput::Bytes(bytes) => {
                self.bytes.extend_from_slice(bytes.as_bytes());
                LoadPoll::NeedInput(self)
            }
            LoadInput::End => {
                match (self.compile)(&LuaString::from(self.bytes), &self.source_name) {
                    Ok(chunk) => LoadPoll::Loaded(chunk),
                    Err(error) => LoadPoll::Failed(LuaString::from(error)),
                }
            }
        }
    }
}

fn set_function_loader(vm: &mut Vm, loader: fn(&LuaString, &LuaString) -> Result<Chunk, String>) {
    vm.set_chunk_loader(Box::new(FunctionLoader(loader)));
}

fn package_tables(runtime: &Runtime) -> (TableHandle, TableHandle) {
    let Value::Table(package) = runtime.get_global("package") else {
        panic!("package is a table");
    };
    let Value::Table(loaded) = runtime.table_get(package, &Value::from("loaded")).unwrap() else {
        panic!("package.loaded is a table");
    };
    let Value::Table(preload) = runtime.table_get(package, &Value::from("preload")).unwrap() else {
        panic!("package.preload is a table");
    };
    (loaded, preload)
}

fn require_chunk(name: LuaString) -> Chunk {
    Chunk::new(
        2,
        vec![Constant::String("require".into()), Constant::String(name)],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
    )
}

fn module_chunk(name: LuaString) -> Chunk {
    Chunk::new(
        2,
        vec![Constant::String("module".into()), Constant::String(name)],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::halt(),
        ],
    )
}

#[derive(Default)]
struct LoaderTrace {
    names: Vec<LuaString>,
    inputs: Vec<LoadInput>,
}

struct TracingLoader {
    trace: Arc<Mutex<LoaderTrace>>,
    chunk: Chunk,
    fail_after_bytes: Option<usize>,
    need_input_after_end: bool,
}

impl TracingLoader {
    fn new(trace: Arc<Mutex<LoaderTrace>>, chunk: Chunk) -> Self {
        Self {
            trace,
            chunk,
            fail_after_bytes: None,
            need_input_after_end: false,
        }
    }

    fn failing_after_bytes(trace: Arc<Mutex<LoaderTrace>>, chunk: Chunk, count: usize) -> Self {
        Self {
            trace,
            chunk,
            fail_after_bytes: Some(count),
            need_input_after_end: false,
        }
    }

    fn needing_input_after_end(trace: Arc<Mutex<LoaderTrace>>, chunk: Chunk) -> Self {
        Self {
            trace,
            chunk,
            fail_after_bytes: None,
            need_input_after_end: true,
        }
    }
}

impl ChunkLoader for TracingLoader {
    fn begin(&self, source_name: LuaString) -> Box<dyn ChunkLoad> {
        self.trace.lock().unwrap().names.push(source_name);
        Box::new(TracingLoad {
            trace: Arc::clone(&self.trace),
            chunk: self.chunk.clone(),
            fail_after_bytes: self.fail_after_bytes,
            need_input_after_end: self.need_input_after_end,
            byte_calls: 0,
        })
    }
}

struct TracingLoad {
    trace: Arc<Mutex<LoaderTrace>>,
    chunk: Chunk,
    fail_after_bytes: Option<usize>,
    need_input_after_end: bool,
    byte_calls: usize,
}

impl ChunkLoad for TracingLoad {
    fn resume(mut self: Box<Self>, input: LoadInput) -> LoadPoll {
        self.trace.lock().unwrap().inputs.push(input.clone());
        match input {
            LoadInput::Bytes(_) => {
                self.byte_calls += 1;
                if self.fail_after_bytes == Some(self.byte_calls) {
                    LoadPoll::Failed(LuaString::from("failed after bytes"))
                } else {
                    LoadPoll::NeedInput(self)
                }
            }
            LoadInput::End if self.need_input_after_end => LoadPoll::NeedInput(self),
            LoadInput::End => LoadPoll::Loaded(self.chunk.clone()),
        }
    }
}

fn reader_loader(source: &LuaString, _: &LuaString) -> Result<Chunk, String> {
    match source.as_bytes() {
        b"answer" => Ok(Chunk::new(
            1,
            vec![Constant::Number(42.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        )),
        b"" => Ok(Chunk::new(
            1,
            vec![Constant::Number(0.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        )),
        [0xff] => Ok(Chunk::new(
            1,
            vec![Constant::Number(9.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        )),
        _ => Err("syntax error".to_owned()),
    }
}

fn sequence_reader(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    let Value::Number(index) = runtime.get_global("reader_index") else {
        return Err(RuntimeError::message("reader index is not a number"));
    };
    let Value::Table(parts) = runtime.get_global("reader_parts") else {
        return Err(RuntimeError::message("reader parts are not a table"));
    };
    runtime.set_global("reader_index", Value::Number(index + 1.0));
    Ok(vec![runtime.table_get(parts, &Value::Number(index))?])
}

fn failing_reader(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
    Err(RuntimeError::message("reader failed"))
}

fn boolean_reader(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
    if !arguments.is_empty() {
        return Err(RuntimeError::message("reader accepts no arguments"));
    }
    Ok(vec![Value::Boolean(true)])
}

fn install_sequence_reader(vm: &mut Vm, pieces: &[Value]) {
    let table = vm.runtime_mut().create_table().unwrap();
    for (offset, piece) in pieces.iter().enumerate() {
        vm.runtime_mut()
            .table_set(
                table,
                &Value::Number(crate::vm::arith::usize_to_number(offset + 1).unwrap()),
                piece.clone(),
            )
            .unwrap();
    }
    vm.runtime_mut()
        .set_global("reader_parts", Value::Table(table));
    vm.runtime_mut()
        .set_global("reader_index", Value::Number(1.0));
    vm.runtime_mut()
        .register_global_native("reader", sequence_reader)
        .unwrap();
}

fn install_getinfo(vm: &mut Vm) {
    vm.runtime_mut().set_global(
        "getinfo",
        Value::BuiltinFunction(BuiltinFunction::DebugGetInfo),
    );
}

fn call_closure(vm: &mut Vm, closure: Value) -> Vec<Value> {
    vm.runtime_mut().set_global("test_closure", closure);
    let caller = Chunk::new(
        1,
        vec![Constant::String("test_closure".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    vm.execute(&caller).unwrap()
}

fn numeric_sum_chunk(start: f64, limit: f64, step: f64) -> Chunk {
    Chunk::new(
        5,
        vec![
            Constant::Number(start),
            Constant::Number(limit),
            Constant::Number(step),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(4, 3),
            Instruction::numeric_for_prep(0, 1),
            Instruction::binary(OpCode::Add, 4, 4, 3),
            Instruction::numeric_for_loop(0, -2),
            Instruction::return_(4),
        ],
    )
}

mod calls;
mod core;
mod coroutines;
mod debug;
mod fast_arith;
mod fast_builtins;
mod fast_coverage;
mod fast_operators;
mod fused_add;
mod host_entries;
mod jit;
mod load;
mod native_boundary;
mod protected_errors;
mod strings;
mod table_gc;
mod trace_cache;
mod trace_callee_guards;
mod trace_calls;
mod trace_cells;
mod trace_deopt;
mod trace_detached;
mod trace_detached_links;
mod trace_integer;
mod trace_link;
mod trace_native;
mod trace_nested;
mod trace_object_keys;
mod trace_record;
mod trace_sides;
mod trace_slot_facts;
mod trace_tables;
mod trace_virtual;
