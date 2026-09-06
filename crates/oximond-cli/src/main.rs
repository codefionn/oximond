#![cfg_attr(not(feature = "alloc-stats"), deny(unsafe_code))]
// On Linux the C entry point in `stack.rs` replaces std's, see there.
#![cfg_attr(all(not(test), target_os = "linux"), no_main)]

#[cfg(feature = "alloc-stats")]
mod alloc;
#[cfg(all(feature = "alloc-stats", feature = "small-blocks"))]
compile_error!(
    "alloc-stats installs its own global allocator; build it with --no-default-features"
);
#[cfg(feature = "alloc-phases")]
mod alloc_phases;
#[cfg(all(not(test), target_os = "linux", target_env = "gnu"))]
mod libm;
mod repl;
mod stack;

use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(all(not(test), not(target_os = "linux")))]
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use oximond::Lua;
use oximond::bytecode::load::{
    LoadPoll, load_file_reader, load_file_reader_pipelined, load_small_file_reader,
};
use oximond::bytecode::{Chunk, binary};
use oximond::compat::CompatibilityProfile;
use oximond::compat::ExecutionMode;
use oximond::compat::{TracingOptFlags, TracingOptLevel};
use oximond::compiler::{CompilerConfig, CompilerLoader, compile_named};
use oximond::runtime::alloc_phase::{Phase, enter as enter_phase};
use oximond::runtime::{ExitRequest, RuntimeError, Value};
use oximond::stdlib::set_package_path;
use oximond::string::LuaString;
use oximond::vm::{TraceAbortReason, TraceExitKind, TraceGuardKind, Vm, VmError};

const USAGE_EXIT: u8 = 64;
const IO_EXIT: u8 = 66;
const COMPILE_EXIT: u8 = 65;
const RUNTIME_EXIT: u8 = 70;
/// Stack for the thread that parses, compiles, and runs Lua: the main thread
/// when its stack limit can be raised to this, else a spawned thread. The
/// streaming parser recurses once per syntax level, and unoptimized builds
/// spend over 100 KiB per level, so Lua's 200-level limit needs about 30 MiB
/// there. The default 8 MiB stack overflows before the limit is reported.
#[cfg(not(test))]
const INTERPRETER_STACK_BYTES: usize = 64 * 1024 * 1024;

static INTERRUPT_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

#[cfg(all(unix, not(test)))]
fn install_interrupt_handler() -> io::Result<()> {
    let flag = Arc::new(AtomicBool::new(false));
    // A second signal stops a native routine that has not returned to the VM.
    // Register the default action first so the first signal only sets the flag.
    signal_hook::flag::register_conditional_default(
        signal_hook::consts::SIGINT,
        Arc::clone(&flag),
    )?;
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&flag))?;
    let _ = INTERRUPT_FLAG.set(flag);
    Ok(())
}

#[cfg(all(not(test), not(target_os = "linux")))]
fn main() -> ExitCode {
    ExitCode::from(u8::try_from(real_main_code()).unwrap_or(1))
}

/// The process body, returning its exit status.
#[cfg(not(test))]
fn real_main_code() -> i32 {
    #[cfg(unix)]
    if let Err(error) = install_interrupt_handler() {
        eprintln!("oximond: cannot install interrupt handler: {error}");
        return i32::from(IO_EXIT);
    }
    let arguments = env::args_os().skip(1).collect();
    let result = if stack::main_is_deep_enough(INTERPRETER_STACK_BYTES) {
        #[cfg(feature = "alloc-phases")]
        alloc_phases::mark_vm_thread();
        run(arguments)
    } else {
        run_on_interpreter_thread(arguments)
    };
    match result {
        Ok(()) => 0,
        Err(error @ (CliError::Exit(_) | CliError::Reported(_))) => i32::from(error.exit_code()),
        Err(error) => {
            eprintln!("oximond: {error}");
            i32::from(error.exit_code())
        }
    }
}

/// Runs on a spawned thread with `INTERPRETER_STACK_BYTES` of stack, for
/// when the main thread's stack cannot be made that deep.
#[cfg(not(test))]
fn run_on_interpreter_thread(arguments: Vec<OsString>) -> Result<(), CliError> {
    let interpreter = std::thread::Builder::new()
        .name("oximond".to_owned())
        .stack_size(INTERPRETER_STACK_BYTES)
        .spawn(move || {
            #[cfg(feature = "alloc-phases")]
            alloc_phases::mark_vm_thread();
            run(arguments)
        });
    match interpreter {
        Ok(handle) => handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
        Err(error) => Err(CliError::Runtime(
            RuntimeError::message(format!("cannot start interpreter thread: {error}")).into(),
        )),
    }
}

#[cfg(not(test))]
fn run(arguments: Vec<OsString>) -> Result<(), CliError> {
    run_with_streaming_writers(arguments, io::stdout(), io::stderr())
}

#[cfg(test)]
fn run_with_writer<W: Write>(arguments: Vec<OsString>, writer: &mut W) -> Result<(), CliError> {
    run_with_writers(arguments, writer, &mut io::sink())
}

#[cfg(test)]
fn run_with_writers<O: Write, E: Write>(
    arguments: Vec<OsString>,
    output: &mut O,
    error_output: &mut E,
) -> Result<(), CliError> {
    run_with_runtime(
        arguments,
        |_| {},
        |vm| {
            output
                .write_all(&vm.runtime_mut().take_output())
                .and_then(|()| output.flush())
                .map_err(|source| CliError::Write { source })?;
            error_output
                .write_all(&vm.runtime_mut().take_error_output())
                .and_then(|()| error_output.flush())
                .map_err(|source| CliError::Write { source })
        },
    )
}

fn run_with_streaming_writers<O: Write + 'static, E: Write + 'static>(
    arguments: Vec<OsString>,
    output: O,
    error_output: E,
) -> Result<(), CliError> {
    run_with_runtime(
        arguments,
        move |vm| {
            vm.runtime_mut().set_output_writer(output);
            vm.runtime_mut().set_error_output_writer(error_output);
        },
        |vm| {
            vm.runtime_mut()
                .flush_output()
                .map_err(|error| CliError::Write {
                    source: io::Error::other(error.to_string()),
                })?;
            vm.runtime_mut()
                .flush_error_output()
                .map_err(|error| CliError::Write {
                    source: io::Error::other(error.to_string()),
                })
        },
    )
}

fn run_with_runtime<C, F>(
    arguments: Vec<OsString>,
    configure_output: C,
    mut finish_output: F,
) -> Result<(), CliError>
where
    C: FnOnce(&mut Vm),
    F: FnMut(&mut Vm) -> Result<(), CliError>,
{
    let plan = parse_plan(arguments)?;
    if plan_has_no_code(&plan) {
        return Ok(());
    }
    let lua = Lua::builder()
        .profile(plan.compatibility)
        .execution_mode(plan.execution_mode)
        .tracing_opt_level(plan.opt_level);
    let lua = match plan.opt_flags {
        Some(flags) => lua.tracing_opt_flags(flags),
        None => lua,
    };
    let lua = lua.build().map_err(|error| match error {
        oximond::Error::Runtime(error) => CliError::Runtime(error),
        oximond::Error::Compile(error) => CliError::Compile(error),
    })?;
    let mut lua =
        oximond_capi::NativeLua::new(lua).map_err(|error| CliError::Runtime(error.into()))?;
    let vm = lua.vm_mut();
    if let Some(flag) = INTERRUPT_FLAG.get() {
        vm.set_interrupt_flag(Arc::clone(flag));
    }
    let profiling_interpreter = env::var_os("OXIMOND_INTERP_PROFILE").is_some();
    vm.set_interpreter_profiling(profiling_interpreter);
    configure_output(vm);
    if let Some(path) = &plan.trace_profile_load {
        let bytes = fs::read(path).map_err(|source| CliError::Read {
            path: path.clone(),
            source,
        })?;
        vm.load_trace_profile(&bytes)
            .map_err(|source| CliError::Read {
                path: path.clone(),
                source,
            })?;
    }
    if plan.trace_profile_record.is_some() {
        vm.start_trace_profile_recording();
    }
    let record_path = plan.trace_profile_record.clone();
    let execution = enter_phase(Phase::Execution);
    let outcome = match run_plan(vm, plan) {
        // `os.exit(code, true)` closes the state first, as `lua_close` does.
        // A finalizer may request its own exit status.
        Err(CliError::Exit(request)) if request.close => {
            Err(CliError::Exit(vm.run_close_finalizers().unwrap_or(request)))
        }
        // The standalone interpreter closes the state after a normal end or
        // an error as well. `os.exit` without `close` leaves the finalizers
        // unrun.
        outcome @ Err(CliError::Exit(_)) => outcome,
        outcome if !vm.runtime().has_finalizers() => outcome,
        // `lua.c` reports a failed script before `lua_close` runs the
        // finalizers, so the error line comes first.
        // A failed flush still lets the finalizers run, and its error is
        // reported after them.
        Err(error) => {
            let flushed = finish_output(vm);
            let line = format!("oximond: {error}\n");
            let reported = vm.runtime_mut().write_error_output(line.as_bytes()).is_ok();
            let request = vm.run_close_finalizers();
            match (flushed, request) {
                (Err(write), _) => Err(write),
                (Ok(()), Some(request)) => Err(CliError::Exit(request)),
                (Ok(()), None) if reported => Err(CliError::Reported(Box::new(error))),
                (Ok(()), None) => Err(error),
            }
        }
        outcome => match vm.run_close_finalizers() {
            Some(request) => Err(CliError::Exit(request)),
            None => outcome,
        },
    };
    let output_result = finish_output(vm);
    execution.end();
    // `OXIMOND_ATTRIBUTION` reports are otherwise written once a second.
    oximond::vm::attribution::flush();
    if let Some(path) = record_path {
        let bytes = vm.trace_profile().map_err(|source| CliError::WriteFile {
            path: path.clone(),
            source,
        })?;
        fs::write(&path, bytes).map_err(|source| CliError::WriteFile { path, source })?;
    }
    output_result?;
    outcome?;
    let _report = enter_phase(Phase::Report);
    if env::var_os("OXIMOND_JIT_STATS").is_some() {
        let statistics = vm.jit_statistics();
        eprintln!(
            "jit: fast_backedge_lookups={} fast_backedge_resumes={}",
            statistics.fast_backedge_lookups, statistics.fast_backedge_resumes,
        );
        eprintln!(
            "interp: inline_calls={} inline_table_calls={} ipairs_steps={}",
            statistics.interpreter_inline_calls,
            statistics.interpreter_inline_table_calls,
            statistics.interpreter_ipairs_steps,
        );
        eprintln!("trace: {:?}", statistics.trace);
        let mir = oximond::jit::mir_counters();
        eprintln!(
            "mir: accepted_trees={} accepted_ir={} fallback_trees={} fallback_ir={} reasons={:?}",
            mir.accepted_trees, mir.accepted_ir, mir.fallback_trees, mir.fallback_ir, mir.reasons,
        );
    }
    if env::var_os("OXIMOND_INTERP_PROFILE").is_some() {
        report_interpreter_profile(vm);
    }
    if env::var_os("OXIMOND_COUNTERS").is_some() {
        eprintln!("{}", counters_line(vm));
    }
    #[cfg(feature = "op-stats")]
    if let Some(path) = env::var_os("OXIMOND_OP_STATS") {
        write_op_stats(vm, Path::new(&path))?;
    }
    // Last, so the report phase covers the reports above.
    #[cfg(feature = "alloc-phases")]
    if let Some(path) = env::var_os("OXIMOND_ALLOC_PHASES") {
        write_alloc_phases(vm, Path::new(&path))?;
    }
    // The process is about to exit. Freeing every table and string one by
    // one is work nobody sees, unless a native library or an open file
    // needs its teardown. Output is flushed above.
    #[cfg(not(test))]
    if lua.teardown_is_inert() {
        std::mem::forget(lua);
    }
    Ok(())
}

/// `on` when the VM's default broker compiles synchronously. The rule of
/// jit/completions.rs: anything but unset, empty or `0`.
#[cfg(any(feature = "op-stats", feature = "alloc-phases"))]
fn jit_sync_label() -> &'static str {
    if env::var_os("OXIMOND_JIT_SYNC").is_some_and(|v| !v.is_empty() && v != "0") {
        "on"
    } else {
        "off"
    }
}

/// Writes the `alloc-phases` report to `path` for `OXIMOND_ALLOC_PHASES`.
/// The counters are copied first, before anything here allocates, so they
/// stop at that point. `benchmarks/measure.py` reads the report.
#[cfg(feature = "alloc-phases")]
fn write_alloc_phases(vm: &Vm, path: &Path) -> Result<(), CliError> {
    let snapshot = alloc_phases::snapshot();
    let build: &[&str] = if cfg!(feature = "op-stats") {
        &["alloc-stats", "alloc-phases", "op-stats"]
    } else {
        &["alloc-stats", "alloc-phases"]
    };
    let execution = format!("{:?}", vm.execution_mode()).to_lowercase();
    let report = alloc_phases::report_json(
        &snapshot,
        build,
        &[("execution", &execution), ("jit_sync", jit_sync_label())],
    );
    fs::write(path, report + "\n").map_err(|source| CliError::Write { source })
}

/// Writes the `op-stats` report to `path` for `OXIMOND_OP_STATS`.
/// `benchmarks/measure.py` reads it.
#[cfg(feature = "op-stats")]
fn write_op_stats(vm: &Vm, path: &Path) -> Result<(), CliError> {
    let build: &[&str] = if cfg!(feature = "alloc-stats") {
        &["op-stats", "alloc-stats"]
    } else {
        &["op-stats"]
    };
    let execution = format!("{:?}", vm.execution_mode()).to_lowercase();
    let jit_sync = jit_sync_label();
    {
        use oximond::runtime::op_stats::{Counter, add};
        // The VM counts inline plan calls in every build; copy them in so the
        // report covers the calls that push no activation.
        let statistics = vm.jit_statistics();
        add(
            Counter::InlinePlanCalls,
            statistics.interpreter_inline_calls,
        );
        add(
            Counter::InlineTablePlanCalls,
            statistics.interpreter_inline_table_calls,
        );
    }
    let report = oximond::runtime::op_stats::report_json(
        |byte| {
            oximond::bytecode::OpCode::try_from(byte)
                .map_or_else(|_| format!("op{byte}"), |opcode| format!("{opcode:?}"))
        },
        build,
        &[("execution", &execution), ("jit_sync", jit_sync)],
    );
    fs::write(path, report + "\n").map_err(|source| CliError::Write { source })
}

/// Renders the one-line `key=value` report that `OXIMOND_COUNTERS` prints to
/// stderr at a clean exit. `benchmarks/measure.py` parses it, so keys only
/// get added, never renamed.
fn counters_line(vm: &Vm) -> String {
    use std::fmt::Write as _;
    let runtime = vm.runtime();
    let (elided_tables, elided_bytes) = runtime.elided_tables_total();
    let trace = vm.jit_statistics().trace;
    let mut line = format!(
        "oximond-counters: tables_created={} elided_tables={elided_tables} elided_table_bytes={elided_bytes} \
         traces_compiled={} native_entries={} iterations={} recording_starts={} traces_recorded={}",
        runtime.tables_created(),
        trace.traces_compiled,
        trace.native_entries,
        trace.iterations,
        trace.recording_starts,
        trace.traces_recorded,
    );
    let _ = write!(
        line,
        " loads_forwarded={} loads_merged={} loads_hoisted={}",
        trace.optimizer.loads_forwarded,
        trace.optimizer.loads_merged,
        trace.optimizer.loads_hoisted,
    );
    let switches = vm.jit_statistics();
    let _ = write!(
        line,
        " coroutine_direct_resumes={} coroutine_direct_yields={} coroutine_root_snapshots={}",
        switches.coroutine_direct_resumes,
        switches.coroutine_direct_yields,
        switches.coroutine_root_snapshots,
    );
    let _ = write!(
        line,
        " coroutine_trace_resumes={} coroutine_trace_replays={} coroutine_root_visits={}",
        switches.coroutine_trace_resumes,
        switches.coroutine_trace_replays,
        switches.coroutine_root_visits,
    );
    let _ = write!(
        line,
        " coroutine_short_runs={} coroutine_trace_unchecked={}",
        switches.coroutine_short_runs, switches.coroutine_trace_unchecked,
    );
    let _ = write!(line, " host_trace_calls={}", switches.host_trace_calls);
    let _ = write!(
        line,
        " coroutine_in_place_resumes={} coroutine_in_place_exits={}",
        switches.coroutine_in_place_resumes, switches.coroutine_in_place_exits,
    );
    #[cfg(feature = "alloc-stats")]
    {
        let (count, bytes, peak) = alloc::snapshot();
        let _ = write!(
            line,
            " alloc_count={count} alloc_bytes={bytes} alloc_peak_bytes={peak}"
        );
    }
    for reason in TraceAbortReason::ALL {
        let count = trace.aborts(reason);
        if count != 0 {
            let _ = write!(line, " abort.{reason:?}={count}");
        }
    }
    for (name, count) in trace.native_aborts.iter() {
        let _ = write!(line, " abort_native.{name}={count}");
    }
    for kind in TraceExitKind::ALL {
        let count = trace.exits_of(kind);
        if count != 0 {
            let _ = write!(line, " exit.{kind:?}={count}");
        }
    }
    for guard in TraceGuardKind::ALL {
        let count = trace.side_exits_of(guard);
        if count != 0 {
            let _ = write!(line, " side_exit.{guard:?}={count}");
        }
    }
    line
}

/// Runs the startup code, `-e` and `-l` actions, script, and interactive
/// session of `plan`.
fn run_plan(vm: &mut Vm, plan: Plan) -> Result<(), CliError> {
    configure_startup(vm, &plan)?;
    let mut action_error = None;
    for action in &plan.actions {
        let source = match action {
            Action::Expression(source) => Source {
                input: SourceInput::Bytes(source.as_bytes().to_vec()),
                name: "=(command line)".to_owned(),
                argument_name: "-e".to_owned(),
            },
            Action::Library(module) => Source {
                input: SourceInput::Bytes(
                    library_action_source(plan.compatibility, module).into_bytes(),
                ),
                name: "=(command line)".to_owned(),
                argument_name: "-l".to_owned(),
            },
        };
        let chunk = source.compile(plan.compatibility)?;
        if let Err(error) = execute_cli_chunk(vm, &chunk, Vec::new(), plan.compatibility) {
            action_error = Some(error);
            break;
        }
    }
    let execution = if let Some(error) = action_error {
        Err(error)
    } else if let Some(source) = plan.script {
        let chunk = source.compile(plan.compatibility)?;
        let argument_name = source.argument_name;
        install_arg(
            vm,
            &plan.argument_prefix,
            &plan.script_args,
            argument_name.as_bytes(),
        )?;
        execute_cli_chunk(
            vm,
            &chunk,
            plan.script_args.iter().cloned().map(Value::from).collect(),
            plan.compatibility,
        )
    } else {
        Ok(Vec::new())
    };
    let execution = if plan.interactive {
        execution.and_then(|_| repl::run(vm))
    } else {
        execution
    };
    execution.map(|_| ()).map_err(cli_runtime_error)
}

fn library_action_source(profile: CompatibilityProfile, argument: &str) -> String {
    if matches!(
        profile,
        CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
    ) {
        return format!("require({})", lua_quote(argument));
    }
    let (global, module) = if matches!(
        profile,
        CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    ) {
        argument.split_once('=').unwrap_or_else(|| {
            (
                argument.split_once('-').map_or(argument, |(name, _)| name),
                argument,
            )
        })
    } else {
        (argument, argument)
    };
    format!("_G[{}] = require({})", lua_quote(global), lua_quote(module))
}

fn execute_cli_chunk(
    vm: &mut Vm,
    chunk: &Chunk,
    arguments: Vec<Value>,
    profile: CompatibilityProfile,
) -> Result<Vec<Value>, VmError> {
    if !matches!(
        profile,
        CompatibilityProfile::Lua52 | CompatibilityProfile::Lua53
    ) {
        return vm.execute_with_arguments(chunk, arguments);
    }
    // Lua's CLI calls __tostring in its protected message handler, while
    // debug.getinfo can still inspect the function that raised the object.
    let handler = compile_named(
        "=(message handler)",
        "return function(e) local mt = debug.getmetatable(e); if type(e) ~= 'string' and mt and mt.__tostring then local message = tostring(e); return message end; return e end",
        CompilerConfig::new(profile),
    ).map_err(|error| RuntimeError::message(error.to_string()))?;
    let values = vm.execute(&handler)?;
    vm.execute_with_message_handler(chunk, arguments, values[0].clone())
}

/// Prints the VM's interpreter hot-path counters for `OXIMOND_INTERP_PROFILE`.
fn report_interpreter_profile(vm: &Vm) {
    let counters = vm.interpreter_counters();
    eprintln!("interp: steps={}", counters.steps.get());
    let mut histogram: Vec<(usize, u64)> = counters
        .opcode_steps()
        .into_iter()
        .enumerate()
        .filter(|&(_, count)| count > 0)
        .collect();
    histogram.sort_by_key(|&(_, count)| u64::MAX - count);
    for (opcode, count) in histogram {
        eprintln!("  opcode {opcode:2}: {count}");
    }
    eprintln!(
        "  tail_fixed: attempts={} fast={} miss_observed={} miss_callee={} miss_vararg={} generic={}",
        counters.tail_fixed_attempts.get(),
        counters.tail_fixed_fast.get(),
        counters.tail_fixed_miss_observed.get(),
        counters.tail_fixed_miss_callee.get(),
        counters.tail_fixed_miss_vararg.get(),
        counters.tail_generic_entries.get()
    );
    eprintln!(
        "  globals={} upvalues={} closure_bodies={} scalar_fast={} scalar_slow={}",
        counters.get_global_lookups.get(),
        counters.get_upvalue_reads.get(),
        counters.closure_body_calls.get(),
        counters.scalar_fast.get(),
        counters.scalar_slow.get()
    );
}

fn configure_package_path(vm: &mut Vm, path: Option<OsString>) -> Result<(), CliError> {
    let Some(path) = path else {
        return Ok(());
    };
    set_package_path(vm.runtime_mut(), path.to_string_lossy().into_owned())
        .map_err(|error| CliError::Runtime(error.into()))
}

fn configure_startup(vm: &mut Vm, plan: &Plan) -> Result<(), CliError> {
    if !plan.ignore_environment {
        let cpath = if plan.compatibility == CompatibilityProfile::Lua55 {
            versioned_environment("LUA_CPATH")
        } else {
            env::var_os("LUA_CPATH")
        };
        if let Some(path) = cpath {
            let Value::Table(package) = vm.runtime().get_global("package") else {
                return Err(CliError::Runtime(
                    RuntimeError::message("package table is unavailable").into(),
                ));
            };
            vm.runtime_mut()
                .table_set(
                    package,
                    &Value::from("cpath"),
                    Value::from(path.to_string_lossy().into_owned()),
                )
                .map_err(|error| CliError::Runtime(error.into()))?;
        }
    }
    // Stock lua.c starts with warnings off and `-W` turns them on.
    vm.runtime_mut().set_warnings_enabled(plan.warnings_enabled);
    if plan.compatibility != CompatibilityProfile::Lua55 {
        return configure_package_path(vm, env::var_os("LUA_PATH"));
    }

    if plan.ignore_environment {
        return Ok(());
    }
    configure_package_path(vm, versioned_environment("LUA_PATH"))?;
    if let Some(init) = versioned_environment("LUA_INIT") {
        execute_lua_init(vm, &init)?;
    }
    Ok(())
}

fn versioned_environment(name: &str) -> Option<OsString> {
    env::var_os(format!("{name}_5_5")).or_else(|| env::var_os(name))
}

fn execute_lua_init(vm: &mut Vm, init: &OsStr) -> Result<(), CliError> {
    let init = init.to_string_lossy();
    let source = if let Some(path) = init.strip_prefix('@') {
        source_file(Path::new(path))
    } else {
        Source {
            input: SourceInput::Bytes(init.as_bytes().to_vec()),
            name: "=LUA_INIT_5_5".to_owned(),
            argument_name: "LUA_INIT_5_5".to_owned(),
        }
    };
    let chunk = source.compile(vm.compatibility_profile())?;
    vm.execute(&chunk).map_err(cli_runtime_error)?;
    Ok(())
}

fn cli_runtime_error(error: VmError) -> CliError {
    if let Some(request) = error.exit_request() {
        return CliError::Exit(request);
    }
    let output_error = match &error {
        VmError::Runtime(RuntimeError::HostOutputIo { message, .. })
        | VmError::LocatedRuntime {
            error: RuntimeError::HostOutputIo { message, .. },
            ..
        } => Some(message),
        _ => None,
    };
    match output_error {
        Some(message) => CliError::Write {
            source: io::Error::other(message.clone()),
        },
        None => CliError::Runtime(error),
    }
}

fn compile_source(
    compatibility: CompatibilityProfile,
    name: String,
    source: &[u8],
) -> Result<Chunk, CliError> {
    // Loading the chunk is startup work, even inside guest execution.
    let _startup = enter_phase(Phase::Unclassified);
    let binary_source = source
        .strip_prefix(b"\n")
        .filter(|source| binary::claims(source))
        .or_else(|| binary::claims(source).then_some(source));
    match binary_source {
        Some(source) => binary::load(source).map_err(|error| CliError::Binary(error.to_string())),
        None => compile_named(name, source, CompilerConfig::new(compatibility))
            .map_err(CliError::Compile),
    }
}

enum SourceInput {
    Bytes(Vec<u8>),
    File(PathBuf),
    Stdin,
}

struct Source {
    input: SourceInput,
    name: String,
    argument_name: String,
}

impl Source {
    fn compile(&self, compatibility: CompatibilityProfile) -> Result<Chunk, CliError> {
        let _startup = enter_phase(Phase::Unclassified);
        let loader = CompilerLoader::with_profile(compatibility);
        let name = LuaString::from(self.name.as_str());
        let poll = match &self.input {
            SourceInput::Bytes(bytes) => {
                return compile_source(compatibility, self.name.clone(), bytes);
            }
            SourceInput::File(path) => {
                let reader = fs::File::open(path).map_err(|source| CliError::Read {
                    path: path.clone(),
                    source,
                })?;
                let result = match reader.metadata() {
                    Ok(metadata) if metadata.is_file() && metadata.len() > 8192 => {
                        load_file_reader_pipelined(&loader, &name, reader, None)
                    }
                    Ok(metadata) if metadata.is_file() => {
                        load_small_file_reader(&loader, &name, reader, None)
                    }
                    _ => load_file_reader(&loader, &name, reader, None),
                };
                result.map_err(|source| CliError::Read {
                    path: path.clone(),
                    source,
                })?
            }
            SourceInput::Stdin => load_file_reader(&loader, &name, io::stdin().lock(), None)
                .map_err(|source| CliError::ReadStdin { source })?,
        };
        match poll {
            LoadPoll::Loaded(chunk) => Ok(chunk),
            LoadPoll::Failed(message) => {
                Err(CliError::CompileDiagnostic(message.to_lossy().into_owned()))
            }
            LoadPoll::NeedInput(_) => Err(CliError::CompileDiagnostic(
                "loader requested input after end of source".to_owned(),
            )),
        }
    }
}

fn source_file(path: &Path) -> Source {
    Source {
        input: SourceInput::File(path.to_path_buf()),
        name: format!("@{}", path.display()),
        argument_name: path.to_string_lossy().into_owned(),
    }
}

fn source_stdin() -> Source {
    Source {
        input: SourceInput::Stdin,
        name: "=stdin".to_owned(),
        argument_name: "-".to_owned(),
    }
}

#[cfg(test)]
fn parse_source(arguments: Vec<OsString>) -> Result<Source, CliError> {
    let plan = parse_plan(arguments)?;
    match (plan.script, plan.actions.as_slice()) {
        (Some(source), _) => Ok(source),
        (None, [Action::Expression(source)]) => Ok(Source {
            input: SourceInput::Bytes(source.as_bytes().to_vec()),
            name: "=(command line)".to_owned(),
            argument_name: "-e".to_owned(),
        }),
        (None, _) => Err(CliError::Usage(usage().to_owned())),
    }
}

struct Plan {
    compatibility: CompatibilityProfile,
    execution_mode: ExecutionMode,
    opt_level: TracingOptLevel,
    /// Replaces the level's flags when set.
    opt_flags: Option<TracingOptFlags>,
    trace_profile_record: Option<PathBuf>,
    trace_profile_load: Option<PathBuf>,
    actions: Vec<Action>,
    script: Option<Source>,
    script_args: Vec<String>,
    argument_prefix: Vec<String>,
    interactive: bool,
    ignore_environment: bool,
    warnings_enabled: bool,
}
enum Action {
    Expression(String),
    Library(String),
}

fn plan_has_no_code(plan: &Plan) -> bool {
    plan.trace_profile_record.is_none()
        && plan.trace_profile_load.is_none()
        && !plan.interactive
        && plan
            .actions
            .iter()
            .all(|action| matches!(action, Action::Expression(source) if source.is_empty()))
        && plan.script.as_ref().is_none_or(
            |source| matches!(&source.input, SourceInput::Bytes(bytes) if bytes.is_empty()),
        )
}

#[allow(clippy::too_many_lines)]
fn parse_plan(arguments: Vec<OsString>) -> Result<Plan, CliError> {
    let mut items = arguments.into_iter().peekable();
    let mut actions = Vec::new();
    let mut script = None;
    let mut script_args = Vec::new();
    let mut argument_prefix = Vec::new();
    let mut interactive = false;
    // Set by `-e` and `-v`, which keep a bare command line from starting a
    // session or reading standard input.
    let mut runs_code = false;
    let mut ignore_environment = false;
    let mut warnings_enabled = false;
    let mut compatibility = CompatibilityProfile::default();
    let mut execution_mode = ExecutionMode::default();
    let mut opt_level = TracingOptLevel::default();
    let mut opt_flags = None;
    let mut trace_profile_record = env::var_os("OXIMOND_TRACE_PROFILE_RECORD").map(PathBuf::from);
    let mut trace_profile_load = env::var_os("OXIMOND_TRACE_PROFILE_LOAD").map(PathBuf::from);
    let mut options = true;
    while let Some(item) = items.next() {
        let text = item
            .into_string()
            .map_err(|_| CliError::Usage(usage().to_owned()))?;
        if script.is_some() {
            script_args.push(text);
            continue;
        }
        if options && text == "--" {
            argument_prefix.push(text);
            options = false;
            continue;
        }
        if options && text == "-v" {
            argument_prefix.push(text);
            runs_code = true;
            actions.push(Action::Expression(
                "print(_VERSION or 'Lua 5.1')".to_owned(),
            ));
            continue;
        }
        if options && text == "-i" {
            argument_prefix.push(text);
            interactive = true;
            continue;
        }
        if options && text == "-E" {
            argument_prefix.push(text);
            ignore_environment = true;
            continue;
        }
        if options && text == "-W" {
            argument_prefix.push(text);
            warnings_enabled = true;
            continue;
        }
        if options && (text == "--compat" || text.starts_with("--compat=")) {
            argument_prefix.push(text.clone());
            let value = if let Some(value) = text.strip_prefix("--compat=") {
                if value.is_empty() {
                    return Err(CliError::Usage(format!(
                        "`--compat` needs an argument\\n\\n{}",
                        usage()
                    )));
                }
                value.to_owned()
            } else {
                let value = items.next().ok_or_else(|| {
                    CliError::Usage(format!("`--compat` needs an argument\\n\\n{}", usage()))
                })?;
                let value = value.into_string().map_err(|_| {
                    CliError::Usage(format!(
                        "`--compat` needs a UTF-8 argument\\n\\n{}",
                        usage()
                    ))
                })?;
                argument_prefix.push(value.clone());
                value
            };
            compatibility = value.parse().map_err(|error| {
                CliError::Usage(format!(
                    "{error}; expected one of {}\\n\\n{}",
                    selectors(),
                    usage()
                ))
            })?;
            continue;
        }
        if options && (text == "--execution" || text.starts_with("--execution=")) {
            argument_prefix.push(text.clone());
            let value = if let Some(value) = text.strip_prefix("--execution=") {
                if value.is_empty() {
                    return Err(CliError::Usage(format!(
                        "`--execution` needs an argument\n\n{}",
                        usage()
                    )));
                }
                value.to_owned()
            } else {
                let value = items.next().ok_or_else(|| {
                    CliError::Usage(format!("`--execution` needs an argument\n\n{}", usage()))
                })?;
                let value = value.into_string().map_err(|_| {
                    CliError::Usage(format!(
                        "`--execution` needs a UTF-8 argument\n\n{}",
                        usage()
                    ))
                })?;
                argument_prefix.push(value.clone());
                value
            };
            execution_mode = value.parse().map_err(|error| {
                CliError::Usage(format!(
                    "{error}; expected one of {}\n\n{}",
                    execution_selectors(),
                    usage()
                ))
            })?;
            continue;
        }
        if options
            && (text == "--trace-profile-record"
                || text.starts_with("--trace-profile-record=")
                || text == "--trace-profile-load"
                || text.starts_with("--trace-profile-load="))
        {
            argument_prefix.push(text.clone());
            let recording = text.starts_with("--trace-profile-record");
            let flag = if recording {
                "--trace-profile-record"
            } else {
                "--trace-profile-load"
            };
            let path = if let Some(value) = text.strip_prefix(&format!("{flag}=")) {
                if value.is_empty() {
                    return Err(CliError::Usage(format!(
                        "`{flag}` needs a path\n\n{}",
                        usage()
                    )));
                }
                PathBuf::from(value)
            } else {
                let value = items.next().ok_or_else(|| {
                    CliError::Usage(format!("`{flag}` needs a path\n\n{}", usage()))
                })?;
                argument_prefix.push(value.to_string_lossy().into_owned());
                PathBuf::from(value)
            };
            if recording {
                trace_profile_record = Some(path);
            } else {
                trace_profile_load = Some(path);
            }
            continue;
        }
        if options && (text == "--opt-level" || text.starts_with("--opt-level=")) {
            argument_prefix.push(text.clone());
            let value = if let Some(value) = text.strip_prefix("--opt-level=") {
                if value.is_empty() {
                    return Err(CliError::Usage(format!(
                        "`--opt-level` needs an argument\n\n{}",
                        usage()
                    )));
                }
                value.to_owned()
            } else {
                let value = items.next().ok_or_else(|| {
                    CliError::Usage(format!("`--opt-level` needs an argument\n\n{}", usage()))
                })?;
                let value = value.into_string().map_err(|_| {
                    CliError::Usage(format!(
                        "`--opt-level` needs a UTF-8 argument\n\n{}",
                        usage()
                    ))
                })?;
                argument_prefix.push(value.clone());
                value
            };
            opt_level = value.parse().map_err(|error| {
                CliError::Usage(format!("{error}; expected 0, 1, or 2\n\n{}", usage()))
            })?;
            continue;
        }
        if options && (text == "--opt-flags" || text.starts_with("--opt-flags=")) {
            argument_prefix.push(text.clone());
            let value = if let Some(value) = text.strip_prefix("--opt-flags=") {
                value.to_owned()
            } else {
                let value = items.next().ok_or_else(|| {
                    CliError::Usage(format!("`--opt-flags` needs an argument\n\n{}", usage()))
                })?;
                let value = value.into_string().map_err(|_| {
                    CliError::Usage(format!(
                        "`--opt-flags` needs a UTF-8 argument\n\n{}",
                        usage()
                    ))
                })?;
                argument_prefix.push(value.clone());
                value
            };
            let bits = match value
                .strip_prefix("0x")
                .or_else(|| value.strip_prefix("0X"))
            {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => value.parse(),
            };
            opt_flags = Some(TracingOptFlags::from_bits(bits.map_err(|_| {
                CliError::Usage(format!(
                    "invalid optimization flags `{value}`; expected a 64-bit number\n\n{}",
                    usage()
                ))
            })?));
            continue;
        }
        let (flag, attached) = if options && text.len() > 2 && text.is_char_boundary(2) {
            (&text[..2], Some(text[2..].to_owned()))
        } else {
            (text.as_str(), None)
        };
        if options && matches!(flag, "-e" | "-l") {
            argument_prefix.push(text.clone());
            let value = if let Some(value) = attached {
                value
            } else {
                let value = items.next().ok_or_else(|| {
                    CliError::Usage(format!("`{flag}` needs an argument\n\n{}", usage()))
                })?;
                let value = value.into_string().map_err(|_| {
                    CliError::Usage(format!("`{flag}` needs a UTF-8 argument\n\n{}", usage()))
                })?;
                argument_prefix.push(value.clone());
                value
            };
            actions.push(if flag == "-e" {
                runs_code = true;
                Action::Expression(value)
            } else {
                Action::Library(value)
            });
            continue;
        }
        if options && text.len() > 1 && text.starts_with('-') {
            return Err(CliError::Usage(format!(
                "unrecognized option '{text}'\n\n{}",
                usage()
            )));
        }
        if script.is_none() {
            if text == "-" {
                script = Some(source_stdin());
            } else {
                script = Some(source_file(Path::new(&text)));
            }
        } else {
            script_args.push(text);
        }
    }
    // As stock `lua` does with neither a script nor `-e` or `-v`: a session
    // on a terminal, the script on standard input otherwise.
    if script.is_none() && !runs_code && !interactive {
        if repl::stdin_is_terminal() {
            interactive = true;
        } else {
            script = Some(source_stdin());
        }
    }
    if (trace_profile_record.is_some() || trace_profile_load.is_some())
        && execution_mode != ExecutionMode::Tracing
    {
        return Err(CliError::Usage(format!(
            "trace profiles require `--execution tracing`\n\n{}",
            usage()
        )));
    }
    Ok(Plan {
        compatibility,
        execution_mode,
        opt_level,
        opt_flags,
        trace_profile_record,
        trace_profile_load,
        actions,
        script,
        script_args,
        argument_prefix,
        interactive,
        ignore_environment,
        warnings_enabled,
    })
}

fn install_arg(
    vm: &mut Vm,
    argument_prefix: &[String],
    arguments: &[String],
    script_name: &[u8],
) -> Result<(), CliError> {
    let arg = vm
        .runtime_mut()
        .create_table()
        .map_err(|error| CliError::Runtime(error.into()))?;
    vm.runtime_mut()
        .table_set(arg, &Value::Number(0.0), Value::bytes(script_name.to_vec()))
        .map_err(|error| CliError::Runtime(error.into()))?;
    let program = env::args_os()
        .next()
        .unwrap_or_else(|| OsString::from("oximond"))
        .to_string_lossy()
        .into_owned();
    let mut preceding = Vec::with_capacity(argument_prefix.len() + 1);
    preceding.push(program);
    preceding.extend(argument_prefix.iter().cloned());
    for (index, value) in preceding.iter().enumerate() {
        vm.runtime_mut()
            .table_set(
                arg,
                &Value::Number(-lua_argument_index(preceding.len() - index)?),
                Value::from(value.clone()),
            )
            .map_err(|error| CliError::Runtime(error.into()))?;
    }
    for (index, value) in arguments.iter().enumerate() {
        vm.runtime_mut()
            .table_set(
                arg,
                &Value::Number(lua_argument_index(
                    index.checked_add(1).ok_or_else(argument_count_error)?,
                )?),
                Value::from(value.clone()),
            )
            .map_err(|error| CliError::Runtime(error.into()))?;
    }
    vm.runtime_mut().set_global("arg", Value::Table(arg));
    Ok(())
}

fn lua_argument_index(index: usize) -> Result<f64, CliError> {
    u32::try_from(index)
        .map(f64::from)
        .map_err(|_| argument_count_error())
}

fn argument_count_error() -> CliError {
    CliError::Usage("too many command line arguments\n\n".to_owned() + usage())
}

fn usage() -> &'static str {
    "Usage: oximond [options] [FILE [args...]]\n       oximond [options] - [args...]\n\nWithout FILE, -e, or -v, oximond starts an interactive session on a\nterminal and runs standard input otherwise.\n\nOptions:\n  --compat=PROFILE           select Lua compatibility profile\n  --execution=MODE           select interpreter or tracing execution\n  --trace-profile-record=FILE  write a trace profile after execution\n  --trace-profile-load=FILE    load a trace profile before execution\n  --opt-level=N              tracing optimization level: 0, 1, or 2\n  --opt-flags=BITS           replace the level's optimization flags (hex with 0x)\n  -E                         ignore Lua startup environment variables\n  -W                         enable warnings\n  -e CHUNK                   execute CHUNK\n  -i                         enter interactive mode after running FILE\n  -l NAME                    require NAME\n  -v                         print version\n  --                         stop handling options"
}

fn selectors() -> String {
    CompatibilityProfile::ALL
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn execution_selectors() -> String {
    ExecutionMode::ALL
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn lua_quote(value: &str) -> String {
    let mut quoted = String::from('"');
    for byte in value.bytes() {
        match byte {
            b'"' => quoted.push_str("\\\""),
            b'\\' => quoted.push_str("\\\\"),
            b'\n' => quoted.push_str("\\n"),
            b'\r' => quoted.push_str("\\r"),
            b'\t' => quoted.push_str("\\t"),
            32..=126 => quoted.push(char::from(byte)),
            _ => {
                use std::fmt::Write as _;
                write!(quoted, "\\{byte:03}").expect("writing to a string cannot fail");
            }
        }
    }
    quoted.push('"');
    quoted
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    Read {
        path: PathBuf,
        source: io::Error,
    },
    ReadStdin {
        source: io::Error,
    },
    Write {
        source: io::Error,
    },
    WriteFile {
        path: PathBuf,
        source: io::Error,
    },
    Compile(oximond::compiler::CompileError),
    CompileDiagnostic(String),
    Binary(String),
    Runtime(oximond::vm::VmError),
    /// Lua called `os.exit`. The process exits with its status, silently.
    Exit(ExitRequest),
    /// An error already written to the error output, before the finalizers
    /// ran. The process exits with its status without printing it again.
    Reported(Box<CliError>),
}

impl CliError {
    const fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) => USAGE_EXIT,
            Self::Read { .. }
            | Self::ReadStdin { .. }
            | Self::Write { .. }
            | Self::WriteFile { .. } => IO_EXIT,
            Self::Compile(_) | Self::CompileDiagnostic(_) | Self::Binary(_) => COMPILE_EXIT,
            Self::Runtime(_) => RUNTIME_EXIT,
            // A process status keeps only the low 8 bits of C's `exit`.
            Self::Exit(request) => request.status.to_le_bytes()[0],
            Self::Reported(error) => error.exit_code(),
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) => formatter.write_str(message),
            Self::Read { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::ReadStdin { source } => write!(formatter, "cannot read standard input: {source}"),
            Self::Write { source } => write!(formatter, "cannot write standard output: {source}"),
            Self::WriteFile { path, source } => {
                write!(formatter, "cannot write {}: {source}", path.display())
            }
            Self::Compile(error) => write!(formatter, "compile error: {error}"),
            Self::CompileDiagnostic(message) => write!(formatter, "compile error: {message}"),
            Self::Binary(error) => write!(formatter, "binary chunk error: {error}"),
            Self::Runtime(error) => write!(formatter, "runtime error: {error}"),
            Self::Exit(request) => write!(formatter, "exit status {}", request.status),
            Self::Reported(error) => error.fmt(formatter),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};
    use std::path::PathBuf;

    use oximond::compat::{ExecutionMode, TracingOptFlags, TracingOptLevel};
    use oximond::runtime::Value;
    use oximond::stdlib::install_base;
    use oximond::vm::Vm;

    use super::{
        Action, CliError, IO_EXIT, Plan, RUNTIME_EXIT, Source, SourceInput, configure_package_path,
        parse_plan, parse_source, plan_has_no_code, run_with_streaming_writers, run_with_writer,
        run_with_writers,
    };

    #[test]
    fn empty_non_interactive_plans_do_not_construct_a_lua_runtime() {
        let plan = Plan {
            compatibility: oximond::compat::CompatibilityProfile::Lua51,
            execution_mode: ExecutionMode::Tracing,
            opt_level: TracingOptLevel::O0,
            opt_flags: None,
            trace_profile_record: None,
            trace_profile_load: None,
            actions: vec![Action::Expression(String::new())],
            script: Some(Source {
                input: SourceInput::Bytes(Vec::new()),
                name: "@empty.lua".to_owned(),
                argument_name: "empty.lua".to_owned(),
            }),
            script_args: vec!["argument".to_owned()],
            argument_prefix: Vec::new(),
            interactive: false,
            ignore_environment: false,
            warnings_enabled: false,
        };

        assert!(plan_has_no_code(&plan));
    }

    #[test]
    fn source_names_follow_lua_chunk_conventions() {
        let inline = parse_source(vec!["-e".into(), "return 42".into()]).unwrap();
        assert_eq!(inline.name, "=(command line)");
        assert!(matches!(inline.input, SourceInput::Bytes(bytes) if bytes == b"return 42"));

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/assignment.lua");
        let expected_name = format!("@{}", path.display());
        let file = parse_source(vec![path.into_os_string()]).unwrap();
        assert_eq!(file.name, expected_name);
    }

    #[test]
    fn cli_compile_errors_include_lua_source_line_and_near_token() {
        let error = run_with_writer(
            vec!["--compat=lua51".into(), "-e".into(), "syntax error".into()],
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CliError::Compile(_)));
        assert_eq!(
            error.to_string(),
            "compile error: (command line):1: expected assignment or function call near 'error'"
        );
    }

    #[test]
    fn planning_a_file_does_not_open_or_read_it() {
        let plan = parse_plan(vec!["/missing/streaming-script.lua".into()]).unwrap();
        assert!(matches!(plan.script.unwrap().input, SourceInput::File(_)));
    }

    #[test]
    fn supplied_lua_path_replaces_the_default_package_path() {
        let mut vm = Vm::new();
        install_base(vm.runtime_mut()).unwrap();
        configure_package_path(&mut vm, Some("./modules/?.lua".into())).unwrap();
        let Value::Table(package) = vm.runtime().get_global("package") else {
            panic!("package is a table");
        };
        assert_eq!(
            vm.runtime()
                .table_get(package, &Value::from("path"))
                .unwrap(),
            Value::from("./modules/?.lua")
        );
    }

    #[test]
    fn cli_preserves_main_and_dynamic_chunk_names_for_debug_info() {
        let mut output = Vec::new();
        run_with_writer(
            vec![
                "-e".into(),
                "local f = function () end; print(debug.getinfo(f).source)".into(),
            ],
            &mut output,
        )
        .unwrap();
        assert_eq!(output, b"=(command line)\n");

        output.clear();
        run_with_writer(
            vec![
                "-e".into(),
                "local loader = assert(load('return function () end', 'named chunk')); print(debug.getinfo(loader()).source)".into(),
            ],
            &mut output,
        )
        .unwrap();
        assert_eq!(output, b"named chunk\n");
    }

    #[test]
    fn cli_writes_print_output_to_its_writer() {
        let mut output = Vec::new();
        run_with_writer(
            vec!["-e".into(), "print('left', true, 2)".into()],
            &mut output,
        )
        .unwrap();
        assert_eq!(output, b"left\ttrue\t2\n");
    }

    #[test]
    fn compatibility_profile_accepts_attached_and_separate_selectors() {
        let attached = parse_plan(vec!["--compat=lua-5.4".into(), "-e".into(), "".into()]).unwrap();
        assert_eq!(attached.compatibility.selector(), "lua54");

        let separate = parse_plan(vec![
            "--compat".into(),
            "lua5.5.1".into(),
            "-e".into(),
            "".into(),
        ])
        .unwrap();
        assert_eq!(separate.compatibility.selector(), "lua55");
    }

    #[test]
    fn lua55_startup_options_record_environment_and_warning_policy() {
        let plan = parse_plan(vec![
            "--compat=lua55".into(),
            "-E".into(),
            "-W".into(),
            "-e".into(),
            "".into(),
        ])
        .unwrap();

        assert!(plan.ignore_environment);
        assert!(plan.warnings_enabled);
    }

    #[test]
    fn compatibility_profile_controls_version_identity() {
        let mut output = Vec::new();
        run_with_writer(
            vec![
                "--compat=lua54".into(),
                "-e".into(),
                "print(_VERSION)".into(),
            ],
            &mut output,
        )
        .unwrap();
        assert_eq!(output, b"Lua 5.4\n");

        output.clear();
        run_with_writer(vec!["-e".into(), "print(_VERSION)".into()], &mut output).unwrap();
        assert_eq!(output, b"Lua 5.1\n");
    }

    #[test]
    fn compatibility_profile_rejects_unknown_selectors_as_usage_errors() {
        let Err(error) = parse_plan(vec!["--compat=lua56".into(), "-e".into(), "".into()]) else {
            panic!("unknown compatibility profile was accepted")
        };
        assert!(matches!(error, CliError::Usage(_)));
        assert_eq!(error.exit_code(), super::USAGE_EXIT);
        assert!(error.to_string().contains("lua51"));
    }

    #[test]
    fn execution_mode_accepts_attached_and_separate_selectors() {
        let attached =
            parse_plan(vec!["--execution=bytecode".into(), "-e".into(), "".into()]).unwrap();
        assert_eq!(attached.execution_mode, ExecutionMode::Interpreter);

        let separate = parse_plan(vec![
            "--execution".into(),
            "tracing".into(),
            "-e".into(),
            "".into(),
        ])
        .unwrap();
        assert_eq!(separate.execution_mode, ExecutionMode::Tracing);
    }

    #[test]
    fn execution_mode_rejects_unknown_selectors_as_usage_errors() {
        let Err(error) = parse_plan(vec!["--execution=mixed".into(), "-e".into(), "".into()])
        else {
            panic!("unknown execution mode was accepted")
        };
        assert!(matches!(error, CliError::Usage(_)));
        assert_eq!(error.exit_code(), super::USAGE_EXIT);
        assert!(error.to_string().contains("interpreter, tracing"));
    }

    #[test]
    fn trace_profile_flags_require_tracing_and_accept_both_path_forms() {
        let record = parse_plan(vec![
            "--execution=tracing".into(),
            "--trace-profile-record=record.bin".into(),
            "-e".into(),
            "print(1)".into(),
        ])
        .unwrap();
        assert_eq!(
            record.trace_profile_record,
            Some(PathBuf::from("record.bin"))
        );

        let load = parse_plan(vec![
            "--trace-profile-load".into(),
            "loaded.bin".into(),
            "--execution".into(),
            "tracing".into(),
            "-e".into(),
            "print(1)".into(),
        ])
        .unwrap();
        assert_eq!(load.trace_profile_load, Some(PathBuf::from("loaded.bin")));

        let error = parse_plan(vec![
            "--trace-profile-record=record.bin".into(),
            "-e".into(),
            "print(1)".into(),
        ])
        .err()
        .expect("interpreter mode should reject trace profiles");
        assert!(matches!(error, CliError::Usage(_)));
        assert!(error.to_string().contains("--execution tracing"));
    }

    #[test]
    fn opt_level_defaults_to_zero_and_accepts_both_forms() {
        let default = parse_plan(vec!["-e".into(), "".into()]).unwrap();
        assert_eq!(default.opt_level, TracingOptLevel::O0);

        let attached = parse_plan(vec!["--opt-level=2".into(), "-e".into(), "".into()]).unwrap();
        assert_eq!(attached.opt_level, TracingOptLevel::O2);

        let separate = parse_plan(vec![
            "--opt-level".into(),
            "1".into(),
            "-e".into(),
            "".into(),
        ])
        .unwrap();
        assert_eq!(separate.opt_level, TracingOptLevel::O1);

        let Err(error) = parse_plan(vec!["--opt-level=3".into(), "-e".into(), "".into()]) else {
            panic!("unknown optimization level was accepted")
        };
        assert!(matches!(error, CliError::Usage(_)));
        assert!(error.to_string().contains("expected 0, 1, or 2"));
    }

    #[test]
    fn opt_flags_accept_hex_and_decimal_bits() {
        let default = parse_plan(vec!["-e".into(), "".into()]).unwrap();
        assert_eq!(default.opt_flags, None);

        let hex = parse_plan(vec!["--opt-flags=0x1".into(), "-e".into(), "".into()]).unwrap();
        assert_eq!(hex.opt_flags, Some(TracingOptFlags::HOIST_GUARDS));

        let decimal = parse_plan(vec![
            "--opt-flags".into(),
            "0".into(),
            "-e".into(),
            "".into(),
        ])
        .unwrap();
        assert_eq!(decimal.opt_flags, Some(TracingOptFlags::NONE));

        assert!(parse_plan(vec!["--opt-flags=x".into(), "-e".into(), "".into()]).is_err());
    }

    #[test]
    fn the_removed_jit_engine_option_and_strict_jit_mode_are_usage_errors() {
        for arguments in [
            vec!["--jit-engine=tracing", "-e", ""],
            vec!["--execution=jit", "-e", ""],
        ] {
            let Err(error) = parse_plan(arguments.into_iter().map(Into::into).collect()) else {
                panic!("a removed option was accepted")
            };
            assert!(matches!(error, CliError::Usage(_)));
            assert_eq!(error.exit_code(), super::USAGE_EXIT);
        }
    }

    #[test]
    fn unknown_options_are_usage_errors_not_script_paths() {
        for option in ["-h", "-f", "-ix", "--help"] {
            let Err(error) = parse_plan(vec![option.into()]) else {
                panic!("unknown option `{option}` was accepted")
            };
            assert!(matches!(error, CliError::Usage(_)));
            assert_eq!(error.exit_code(), super::USAGE_EXIT);
            assert!(
                error
                    .to_string()
                    .starts_with(&format!("unrecognized option '{option}'"))
            );
        }
        let plan = parse_plan(vec!["--".into(), "-h".into()]).unwrap();
        let error = plan
            .script
            .unwrap()
            .compile(plan.compatibility)
            .unwrap_err();
        assert!(matches!(error, CliError::Read { .. }));
    }

    #[test]
    fn cli_propagates_execution_mode_into_the_vm() {
        let mut output = Vec::new();
        run_with_writer(
            vec![
                "--compat=lua51".into(),
                "--execution=tracing".into(),
                "-e".into(),
                "print('fallback')".into(),
            ],
            &mut output,
        )
        .unwrap();
        assert_eq!(output, b"fallback\n");
    }

    #[test]
    fn cli_keeps_io_stderr_separate_from_standard_output() {
        let mut output = Vec::new();
        let mut error_output = Vec::new();
        run_with_writers(
            vec!["-e".into(), "io.stderr:write('diagnostic')".into()],
            &mut output,
            &mut error_output,
        )
        .unwrap();
        assert_eq!(output, []);
        assert_eq!(error_output, b"diagnostic");
    }

    const FAILING_WITH_FINALIZER: &str = "setmetatable({}, {__gc = function() \
        io.stderr:write('gc\\n') end}) print('before') error('after')";

    #[test]
    fn cli_reports_a_failed_script_through_the_error_output_before_finalizers() {
        let mut output = Vec::new();
        let mut error_output = Vec::new();
        let error = run_with_writers(
            vec![
                "--compat=lua54".into(),
                "-e".into(),
                FAILING_WITH_FINALIZER.into(),
            ],
            &mut output,
            &mut error_output,
        )
        .unwrap_err();
        let CliError::Reported(reported) = &error else {
            panic!("expected a reported error, got {error:?}");
        };
        assert!(matches!(**reported, CliError::Runtime(_)));
        assert_eq!(error.exit_code(), RUNTIME_EXIT);
        assert_eq!(output, b"before\n");
        assert_eq!(
            String::from_utf8_lossy(&error_output),
            "oximond: runtime error: (command line):1: after\ngc\n"
        );
    }

    #[test]
    fn cli_runs_finalizers_when_flushing_a_failed_script_fails() {
        let mut error_output = Vec::new();
        let error = run_with_writers(
            vec![
                "--compat=lua54".into(),
                "-e".into(),
                FAILING_WITH_FINALIZER.into(),
            ],
            &mut FailingWriter,
            &mut error_output,
        )
        .unwrap_err();
        assert!(matches!(error, CliError::Write { .. }), "{error:?}");
        assert_eq!(error.exit_code(), IO_EXIT);
        assert_eq!(
            String::from_utf8_lossy(&error_output),
            "oximond: runtime error: (command line):1: after\ngc\n"
        );
    }

    #[test]
    fn cli_flushes_printed_output_before_reporting_a_runtime_error() {
        let mut output = Vec::new();
        let error = run_with_writer(
            vec!["-e".into(), "print('before'); error('after')".into()],
            &mut output,
        )
        .unwrap_err();
        assert!(matches!(error, CliError::Runtime(_)));
        assert_eq!(error.exit_code(), RUNTIME_EXIT);
        assert_eq!(output, b"before\n");
    }

    #[test]
    fn cli_reports_output_write_failures() {
        let mut writer = FailingWriter;
        let error =
            run_with_writer(vec!["-e".into(), "print('output')".into()], &mut writer).unwrap_err();
        assert!(matches!(error, CliError::Write { .. }));
        assert_eq!(error.exit_code(), IO_EXIT);
        assert!(error.to_string().contains("cannot write standard output"));
    }

    #[test]
    fn cli_reports_streaming_output_write_failures_as_io_errors() {
        let error = run_with_streaming_writers(
            vec!["-e".into(), "print('output')".into()],
            FailingWriter,
            io::sink(),
        )
        .unwrap_err();
        assert!(matches!(error, CliError::Write { .. }));
        assert_eq!(error.exit_code(), IO_EXIT);
        assert!(error.to_string().contains("cannot write standard output"));
    }

    #[test]
    fn lua_standard_stream_methods_return_host_io_errors() {
        struct FailedStream;

        impl Write for FailedStream {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::from_raw_os_error(5))
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::from_raw_os_error(5))
            }
        }

        let mut vm = Vm::new();
        install_base(vm.runtime_mut()).unwrap();
        vm.runtime_mut().set_output_writer(FailedStream);
        vm.runtime_mut().set_error_output_writer(FailedStream);
        let chunk = super::compile_source(
            vm.compatibility_profile(),
            "=host I/O errors".to_owned(),
            br"
                local function check(value, message, code)
                    assert(value == nil)
                    assert(type(message) == 'string')
                    assert(code == 5)
                end
                check(io.write('x'))
                check(io.flush())
                check(io.stdout:write('x'))
                check(io.stdout:flush())
                check(io.stderr:write('x'))
                check(io.stderr:flush())
            ",
        )
        .unwrap();
        vm.execute(&chunk).unwrap();
    }

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("broken pipe"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
