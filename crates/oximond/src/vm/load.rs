//! `load`, `loadfile`, `dofile`, and `require` pipelines that compile and install chunks.

use crate::bytecode::load::{
    ChunkLoad, ChunkLoader, LoadInput, LoadPoll, load_file_reader, load_file_reader_pipelined,
    load_small_file_reader,
};
use crate::bytecode::{Chunk, Instruction, binary};
use crate::compat::CompatibilityProfile;
use crate::runtime::{
    BuiltinFunction, CellHandle, ClosureHandle, RuntimeError, RuntimeResult, TableHandle, Value,
};
use crate::string::LuaString;
use crate::vm::builtins::builtin_table_value;
use crate::vm::continuations::map_load_poll;
use crate::vm::errors::raised_value;
use crate::vm::registers::write_call_results;
use crate::vm::stack::CallStack;
use crate::vm::{
    BINARY_MAGIC, DofileCompletion, LoadCompletion, LoadPipeline, LoadReaderContinuation,
    MAX_LUA_CALL_FRAMES, ModuleCompletion, ModuleContinuation, PipelinePoll, RequireCompletion,
    RequireContinuation, RequireSearchContinuation, ReturnTarget, Vm, VmError, VmResult,
};
use std::io::{Cursor, Read};
use std::sync::Arc;

/// Binary files do not require a text compiler.
struct MissingSourceLoader;

impl ChunkLoader for MissingSourceLoader {
    fn begin(&self, _: LuaString) -> Box<dyn ChunkLoad> {
        Box::new(Self)
    }
}

impl ChunkLoad for MissingSourceLoader {
    fn resume(self: Box<Self>, _: LoadInput) -> LoadPoll {
        LoadPoll::Failed(LuaString::from("no source loader configured"))
    }
}

/// Only regular native files can safely read ahead of parser failures.
pub(crate) struct HostSource {
    reader: Box<dyn Read>,
    policy: FileLoadPolicy,
}

enum FileLoadPolicy {
    Streaming,
    Complete,
    Pipelined,
}

impl HostSource {
    fn complete(bytes: Vec<u8>) -> Self {
        Self {
            reader: Box::new(Cursor::new(bytes)),
            policy: FileLoadPolicy::Streaming,
        }
    }
}

fn package_string(
    runtime: &crate::runtime::Runtime,
    arguments: &[Value],
    index: usize,
    default: Option<&[u8]>,
) -> RuntimeResult<LuaString> {
    match arguments.get(index) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::Integer(value)) => Ok(LuaString::from(value.to_string())),
        Some(Value::Number(value)) => Ok(runtime.number_format().format_string(*value)),
        None | Some(Value::Nil) if default.is_some() => Ok(LuaString::from(default.unwrap())),
        value => Err(RuntimeError::bad_argument(index + 1, "string", value)),
    }
}

fn replace_bytes(source: &[u8], pattern: &[u8], replacement: &[u8]) -> Vec<u8> {
    if pattern.is_empty() {
        return source.to_vec();
    }
    let mut result = Vec::new();
    let mut cursor = 0;
    while cursor < source.len() {
        if source[cursor..].starts_with(pattern) {
            result.extend_from_slice(replacement);
            cursor += pattern.len();
        } else {
            result.push(source[cursor]);
            cursor += 1;
        }
    }
    result
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "Non-Unix hosts reject paths that cannot be represented as UTF-8."
)]
fn host_path(bytes: &[u8]) -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Some(std::path::PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
    }
    #[cfg(not(unix))]
    {
        std::str::from_utf8(bytes)
            .ok()
            .map(std::path::PathBuf::from)
    }
}

impl Vm {
    /// Searches paths through the same host boundary as `loadfile`.
    pub(crate) fn package_searchpath(&self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let name = package_string(&self.runtime, arguments, 0, None)?;
        let path = package_string(&self.runtime, arguments, 1, None)?;
        let separator = package_string(&self.runtime, arguments, 2, Some(b"."))?;
        #[cfg(windows)]
        let replacement = package_string(&self.runtime, arguments, 3, Some(b"\\"))?;
        #[cfg(not(windows))]
        let replacement = package_string(&self.runtime, arguments, 3, Some(b"/"))?;
        let module = replace_bytes(
            name.as_bytes(),
            separator.as_bytes(),
            replacement.as_bytes(),
        );
        let mut diagnostics = Vec::new();
        for template in path
            .as_bytes()
            .split(|byte| *byte == b';')
            .filter(|part| !part.is_empty())
        {
            let candidate = replace_bytes(template, b"?", &module);
            let readable = if self.source_reader.is_some() {
                self.read_host_source(Some(&candidate)).is_ok()
            } else {
                host_path(&candidate).is_some_and(|path| std::fs::File::open(path).is_ok())
            };
            if readable {
                return Ok(vec![Value::String(LuaString::from(candidate))]);
            }
            diagnostics.extend_from_slice(b"\n\tno file '");
            diagnostics.extend_from_slice(&candidate);
            diagnostics.push(b'\'');
        }
        Ok(vec![
            Value::Nil,
            Value::String(LuaString::from(diagnostics)),
        ])
    }

    pub(crate) fn package_source_searcher(&mut self, arguments: &[Value]) -> VmResult<Vec<Value>> {
        let name = package_string(&self.runtime, arguments, 0, None)?;
        let package = self
            .runtime
            .package_require_table()
            .ok_or_else(|| RuntimeError::message("package table is unavailable"))?;
        let Value::String(path) = self.runtime.table_get(package, &Value::from("path"))? else {
            return Err(RuntimeError::message("'package.path' must be a string").into());
        };
        let resolved = self
            .module_resolver
            .as_ref()
            .map(|resolver| resolver(name.as_bytes()))
            .transpose()
            .map_err(RuntimeError::message)?
            .flatten();
        let (results, source_name) = if let Some((source, source_name)) = resolved {
            let source_name = LuaString::from(source_name);
            (
                self.load_string(&LuaString::from(source), &source_name),
                source_name,
            )
        } else {
            let module = replace_bytes(
                name.as_bytes(),
                b".",
                std::path::MAIN_SEPARATOR_STR.as_bytes(),
            );
            let mut found = None;
            let mut diagnostics = Vec::new();
            for template in path
                .as_bytes()
                .split(|byte| *byte == b';')
                .filter(|part| !part.is_empty())
            {
                let candidate = replace_bytes(template, b"?", &module);
                if let Ok(source) = self.open_host_source(&candidate) {
                    found = Some((candidate, source));
                    break;
                }
                diagnostics.extend_from_slice(b"\n\tno file '");
                diagnostics.extend_from_slice(&candidate);
                diagnostics.push(b'\'');
            }
            let Some((filename, source)) = found else {
                return Ok(vec![Value::String(LuaString::from(diagnostics))]);
            };
            let mut source_name = vec![b'@'];
            source_name.extend_from_slice(&filename);
            let source_name = LuaString::from(source_name);
            let results = self
                .load_file_source(source, &source_name, None)
                .unwrap_or_else(|error| vec![Value::Nil, Value::from(error.to_string())]);
            (results, source_name)
        };
        match results.as_slice() {
            [loader @ Value::Closure(_)] => {
                if self.compatibility.is_lua51_family() {
                    return Ok(vec![loader.clone()]);
                }
                let filename = source_name
                    .as_bytes()
                    .strip_prefix(b"@")
                    .unwrap_or(source_name.as_bytes());
                Ok(vec![
                    loader.clone(),
                    Value::String(LuaString::from(filename)),
                ])
            }
            [Value::Nil, Value::String(error), ..] => Err(RuntimeError::message(format!(
                "error loading module '{}' from file '{}':\n\t{}",
                name.to_lossy(),
                source_name.to_lossy().trim_start_matches('@'),
                error.to_lossy()
            ))
            .into()),
            _ => Err(RuntimeError::message("module loader returned an invalid result").into()),
        }
    }
    pub(crate) fn apply_load_environment(
        &mut self,
        results: Vec<Value>,
        environment: Option<&Value>,
    ) -> Vec<Value> {
        let Some(environment) = environment else {
            return results;
        };
        if let Some(Value::Closure(closure)) = results.first() {
            let cell = match self.closure_body(*closure) {
                Ok(body) => body.cells().first().copied(),
                Err(error) => return vec![Value::Nil, Value::from(error.to_string())],
            };
            if let Some(cell) = cell
                && let Err(error) = self.runtime.cell_set(cell, environment.clone())
            {
                return vec![Value::Nil, Value::from(error.to_string())];
            }
        }
        results
    }

    pub(crate) fn load_string(
        &mut self,
        source: &LuaString,
        source_name: &LuaString,
    ) -> Vec<Value> {
        // Text that is complete up front skips the streaming session, so the
        // loader can parse it in one pass.
        let poll = if binary::claims(source.as_bytes()) {
            let poll = self.resume_load_pipeline(
                LoadPipeline::new(source_name.clone()),
                LoadInput::Bytes(source.clone()),
            );
            match poll {
                PipelinePoll::NeedInput(session) => {
                    self.resume_load_pipeline(session, LoadInput::End)
                }
                terminal => terminal,
            }
        } else {
            match self.chunk_loader.as_deref() {
                Some(loader) => {
                    map_load_poll(loader.load_complete(source_name.clone(), source.clone()))
                }
                None => PipelinePoll::Failed(LuaString::from("no source loader configured")),
            }
        };
        self.finish_load_pipeline(poll)
    }

    pub(crate) fn start_module(
        &mut self,
        frames: &mut CallStack,
        completion: ModuleCompletion,
        arguments: &[Value],
    ) -> VmResult<()> {
        let name = match arguments.first() {
            Some(Value::String(name)) if !name.as_bytes().is_empty() => name.clone(),
            Some(value) => {
                return Err(RuntimeError::TypeError {
                    operation: "call module with",
                    expected: "non-empty string",
                    found: value.type_name(),
                }
                .into());
            }
            None => {
                return Err(RuntimeError::TypeError {
                    operation: "call module with",
                    expected: "non-empty string",
                    found: "nil",
                }
                .into());
            }
        };
        if name
            .as_bytes()
            .split(|byte| *byte == b'.')
            .any(<[u8]>::is_empty)
        {
            return Err(RuntimeError::message("invalid module name").into());
        }

        let package = builtin_table_value(&self.runtime.get_global("package"), "call module with")?;
        let loaded = builtin_table_value(
            &self.runtime.table_get(package, &Value::from("loaded"))?,
            "call module with",
        )?;
        let key = Value::String(name.clone());
        let existing = self.runtime.table_get(loaded, &key)?;
        let replaces_require_sentinel = self.is_require_sentinel(loaded, &name, &existing);
        let module = match existing {
            // Lua 5.1 treats an existing `package.loaded` table as the module
            // table, regardless of the matching global value. The private
            // `require` sentinel is the exception: `module` must replace it
            // with the global module table, both during and after a failed
            // require.
            Value::Table(table) if !replaces_require_sentinel => table,
            // `luaL_findtable` creates or reuses the dotted global table for
            // every non-table `package.loaded[name]` value.
            _ => {
                let table = self.publish_module_global(&name)?;
                self.runtime.table_set(loaded, &key, Value::Table(table))?;
                table
            }
        };
        if replaces_require_sentinel {
            self.require_sentinels.remove(&(loaded, name.clone()));
        }

        // Lua 5.1 initializes a module table once. Re-entering `module` with
        // a table that already has `_NAME` changes the caller environment but
        // leaves the table's public metadata alone.
        if matches!(
            self.runtime.table_get(module, &Value::from("_NAME"))?,
            Value::Nil
        ) {
            self.runtime
                .table_set(module, &Value::from("_M"), Value::Table(module))?;
            self.runtime
                .table_set(module, &Value::from("_NAME"), Value::String(name.clone()))?;
            let package_name = name
                .as_bytes()
                .iter()
                .rposition(|byte| *byte == b'.')
                .map_or_else(Vec::new, |index| name.as_bytes()[..=index].to_vec());
            self.runtime.table_set(
                module,
                &Value::from("_PACKAGE"),
                Value::String(LuaString::from(package_name)),
            )?;
        }

        let mut caller = frames.last_mut().expect("module caller frame exists");
        caller.environment = module;
        if let Some(closure) = caller.closure {
            self.runtime.set_closure_environment(closure, module)?;
        }

        self.advance_module(
            frames,
            ModuleContinuation {
                module,
                options: arguments[1..].to_vec(),
                next_option: 0,
                completion,
            },
        )
    }

    pub(crate) fn publish_module_global(&mut self, name: &LuaString) -> VmResult<TableHandle> {
        let parts = name
            .as_bytes()
            .split(|byte| *byte == b'.')
            .collect::<Vec<_>>();
        let mut parent = self.runtime.global_environment();
        for part in &parts[..parts.len() - 1] {
            let key = Value::bytes(part.to_vec());
            parent = match self.runtime.table_get(parent, &key)? {
                Value::Nil => {
                    let child = self.runtime.create_table()?;
                    self.runtime.table_set(parent, &key, Value::Table(child))?;
                    child
                }
                Value::Table(child) => child,
                _ => return Err(RuntimeError::message("name conflict for module").into()),
            };
        }
        let final_key = Value::bytes(parts[parts.len() - 1].to_vec());
        let module = match self.runtime.table_get(parent, &final_key)? {
            Value::Nil => {
                let module = self.runtime.create_table()?;
                self.runtime
                    .table_set(parent, &final_key, Value::Table(module))?;
                module
            }
            Value::Table(existing) => existing,
            _ => return Err(RuntimeError::message("name conflict for module").into()),
        };
        Ok(module)
    }

    pub(crate) fn advance_module(
        &mut self,
        frames: &mut CallStack,
        mut continuation: ModuleContinuation,
    ) -> VmResult<()> {
        while let Some(option) = continuation.options.get(continuation.next_option).cloned() {
            continuation.next_option += 1;
            let arguments = [Value::Table(continuation.module)];
            match option {
                Value::NativeFunction(function) => {
                    self.call_native(frames, function, &arguments)?;
                }
                Value::Closure(function) => {
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        arguments.to_vec(),
                        body.shared_cells(),
                        Some(ReturnTarget::Module(Box::new(continuation))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(());
                }
                Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, &arguments) {
                    Err(VmError::InvalidDispatch(_)) => {
                        return self.push_builtin_call_frame(
                            frames,
                            builtin,
                            arguments.to_vec(),
                            ReturnTarget::Module(Box::new(continuation)),
                        );
                    }
                    results => {
                        results?;
                    }
                },
                value => {
                    return Err(RuntimeError::TypeError {
                        operation: "call module option",
                        expected: "function",
                        found: value.type_name(),
                    }
                    .into());
                }
            }
        }
        self.finish_module(frames, continuation.completion)
    }

    pub(crate) fn finish_module(
        &mut self,
        frames: &mut CallStack,
        completion: ModuleCompletion,
    ) -> VmResult<()> {
        let mut caller = frames.last_mut().expect("module caller frame exists");
        match completion {
            ModuleCompletion::Direct(target) => write_call_results(
                &mut caller,
                Instruction::call(target.base, 0, target.results),
                Vec::new(),
            )?,
            ModuleCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut caller,
                    target.base,
                    target.results,
                    true,
                    Vec::new(),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn start_require(
        &mut self,
        frames: &mut CallStack,
        completion: RequireCompletion,
        arguments: &[Value],
    ) -> VmResult<()> {
        let name = match arguments.first() {
            Some(Value::String(name)) => name.clone(),
            argument => return Err(RuntimeError::bad_argument(1, "string", argument).into()),
        };
        let (loaded_table, preload, path, cpath) = self.package_require_state()?;
        let cached = self
            .runtime
            .table_get(loaded_table, &Value::String(name.clone()))?;
        if cached.is_truthy() {
            if self.is_require_sentinel(loaded_table, &name, &cached) {
                return Err(RuntimeError::message(format!(
                    "loop or previous error loading module '{}'",
                    name.to_lossy()
                ))
                .into());
            }
            return self.finish_require(frames, completion, vec![cached]);
        }

        // Lua 5.1 only publishes the sentinel after a searcher has supplied a
        // loader. A missing module or source compilation error is therefore
        // retryable. An error thrown by the selected loader is not.
        let preloaded = self
            .runtime
            .table_get(preload, &Value::String(name.clone()))?;
        let package = self.runtime.package_require_table().or_else(|| {
            match self.runtime.get_global("package") {
                Value::Table(table) => Some(table),
                _ => None,
            }
        });
        if let Some(package) = package
            && let Value::Table(loaders) =
                self.runtime.table_get(package, &Value::from("loaders"))?
        {
            return self.advance_require_search(
                frames,
                RequireSearchContinuation {
                    name,
                    loaded: loaded_table,
                    loaders,
                    next_index: 1,
                    diagnostics: String::new(),
                    completion,
                },
            );
        }
        let module_loader = if matches!(preloaded, Value::Nil) {
            let resolved = self
                .module_resolver
                .as_ref()
                .map(|resolver| resolver(name.as_bytes()))
                .transpose()
                .map_err(RuntimeError::message)?;
            let results = if let Some((source, source_name)) = resolved.flatten() {
                self.load_string(&LuaString::from(source), &LuaString::from(source_name))
            } else {
                let (source, source_name) = self.find_module_source(&path, &cpath, &name)?;
                self.load_file_source(source, &source_name, None)
                    .map_err(|error| RuntimeError::message(error.to_string()))?
            };
            match results.as_slice() {
                [Value::Closure(closure)] => Value::Closure(*closure),
                [Value::Nil, Value::String(error), ..] => {
                    return Err(RuntimeError::message(error.to_lossy()).into());
                }
                _ => {
                    return Err(
                        RuntimeError::message("module loader returned an invalid result").into(),
                    );
                }
            }
        } else {
            preloaded
        };
        let sentinel = self.runtime.create_table()?;
        self.runtime.table_set(
            loaded_table,
            &Value::String(name.clone()),
            Value::Table(sentinel),
        )?;
        let continuation = RequireContinuation {
            loader_data: Value::Nil,
            name: name.clone(),
            loaded: loaded_table,
            sentinel,
            completion,
        };
        self.require_sentinels
            .insert((loaded_table, name.clone()), sentinel);
        self.call_required_loader(frames, continuation.clone(), module_loader, name)?;
        Ok(())
    }

    pub(crate) fn advance_require_search(
        &mut self,
        frames: &mut CallStack,
        mut search: RequireSearchContinuation,
    ) -> VmResult<()> {
        loop {
            let searcher = self
                .runtime
                .table_get(search.loaders, &Value::Number(search.next_index as f64))?;
            if matches!(searcher, Value::Nil) {
                return Err(RuntimeError::message(format!(
                    "module '{}' not found:{}",
                    search.name.to_lossy(),
                    search.diagnostics
                ))
                .into());
            }
            search.next_index += 1;
            let args = [Value::String(search.name.clone())];
            let results = match searcher {
                Value::NativeFunction(function) => self.call_native(frames, function, &args)?,
                Value::BuiltinFunction(function) => match self.call_builtin(function, &args) {
                    Err(VmError::InvalidDispatch(_)) => {
                        return self.push_builtin_call_frame(
                            frames,
                            function,
                            args.to_vec(),
                            ReturnTarget::RequireSearch(Box::new(search)),
                        );
                    }
                    results => results?,
                },
                Value::Closure(function) => {
                    let body = self.closure_body(function)?;
                    let frame = self.make_frame(
                        body.chunk(),
                        args.to_vec(),
                        body.shared_cells(),
                        Some(ReturnTarget::RequireSearch(Box::new(search))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(());
                }
                value => {
                    return Err(RuntimeError::TypeError {
                        operation: "call package loader",
                        expected: "function",
                        found: value.type_name(),
                    }
                    .into());
                }
            };
            match self.finish_require_search_result(frames, search, results)? {
                Some(next) => search = next,
                None => return Ok(()),
            }
        }
    }

    pub(crate) fn resume_require_search(
        &mut self,
        frames: &mut CallStack,
        search: RequireSearchContinuation,
        results: Vec<Value>,
    ) -> VmResult<()> {
        if let Some(next) = self.finish_require_search_result(frames, search, results)? {
            self.advance_require_search(frames, next)
        } else {
            Ok(())
        }
    }

    // The require-search continuations pass `Vec<Value>` by protocol; a slice
    // here would only move the by-value lint onto their signatures.
    #[allow(clippy::needless_pass_by_value)]
    fn finish_require_search_result(
        &mut self,
        frames: &mut CallStack,
        mut search: RequireSearchContinuation,
        results: Vec<Value>,
    ) -> VmResult<Option<RequireSearchContinuation>> {
        match results.first() {
            Some(Value::NativeFunction(_) | Value::Closure(_) | Value::BuiltinFunction(_)) => {
                let loader = results[0].clone();
                let sentinel = self.runtime.create_table()?;
                self.runtime.table_set(
                    search.loaded,
                    &Value::String(search.name.clone()),
                    Value::Table(sentinel),
                )?;
                self.require_sentinels
                    .insert((search.loaded, search.name.clone()), sentinel);
                self.call_required_loader(
                    frames,
                    RequireContinuation {
                        loader_data: results.get(1).cloned().unwrap_or(Value::Nil),
                        name: search.name.clone(),
                        loaded: search.loaded,
                        sentinel,
                        completion: search.completion,
                    },
                    loader,
                    search.name,
                )?;
                Ok(None)
            }
            Some(Value::String(message)) => {
                search.diagnostics.push_str(&message.to_lossy());
                Ok(Some(search))
            }
            _ => Ok(Some(search)),
        }
    }

    pub(crate) fn package_require_state(
        &self,
    ) -> VmResult<(TableHandle, TableHandle, LuaString, LuaString)> {
        let package = match self.runtime.package_require_table() {
            Some(package) => package,
            // Keep supporting deliberately minimal embeddings that install a
            // `require` builtin without the standard package library.
            None => builtin_table_value(&self.runtime.get_global("package"), "call require with")?,
        };
        let loaded = builtin_table_value(
            &self.runtime.table_get(package, &Value::from("loaded"))?,
            "call require with",
        )?;
        let preload = builtin_table_value(
            &self.runtime.table_get(package, &Value::from("preload"))?,
            "call require with",
        )?;
        let path = match self.runtime.table_get(package, &Value::from("path"))? {
            Value::String(path) => path,
            value => {
                return Err(RuntimeError::TypeError {
                    operation: "call require with",
                    expected: "string package.path",
                    found: value.type_name(),
                }
                .into());
            }
        };
        let cpath = match self.runtime.table_get(package, &Value::from("cpath"))? {
            Value::String(path) => path,
            value => {
                return Err(RuntimeError::TypeError {
                    operation: "call require with",
                    expected: "string package.cpath",
                    found: value.type_name(),
                }
                .into());
            }
        };
        Ok((loaded, preload, path, cpath))
    }

    pub(crate) fn find_module_source(
        &self,
        path: &LuaString,
        cpath: &LuaString,
        name: &LuaString,
    ) -> RuntimeResult<(HostSource, LuaString)> {
        let module_path = name
            .as_bytes()
            .iter()
            .map(|byte| if *byte == b'.' { b'/' } else { *byte })
            .collect::<Vec<_>>();
        let mut diagnostics = format!(
            "module '{}' not found:\n\tno field package.preload['{}']",
            name.to_lossy(),
            name.to_lossy()
        );
        for template in path.as_bytes().split(|byte| *byte == b';') {
            if template.is_empty() {
                continue;
            }
            let mut candidate = Vec::new();
            for byte in template {
                if *byte == b'?' {
                    candidate.extend_from_slice(&module_path);
                } else {
                    candidate.push(*byte);
                }
            }
            let display = String::from_utf8_lossy(&candidate);
            if let Ok(source) = self.open_host_source(&candidate) {
                let mut source_name = Vec::with_capacity(candidate.len() + 1);
                source_name.push(b'@');
                source_name.extend_from_slice(&candidate);
                return Ok((source, LuaString::from(source_name)));
            }
            diagnostics.push_str("\n\tno file '");
            diagnostics.push_str(&display);
            diagnostics.push('\'');
        }
        for template in cpath.as_bytes().split(|byte| *byte == b';') {
            if template.is_empty() {
                continue;
            }
            let mut candidate = Vec::new();
            for byte in template {
                if *byte == b'?' {
                    candidate.extend_from_slice(&module_path);
                } else {
                    candidate.push(*byte);
                }
            }
            diagnostics.push_str("\n\tno file '");
            diagnostics.push_str(&String::from_utf8_lossy(&candidate));
            diagnostics.push('\'');
        }
        diagnostics.push_str("\n\tnative module loading is unavailable in this Rust-only build");
        Err(RuntimeError::message(diagnostics))
    }

    pub(crate) fn call_required_loader(
        &mut self,
        frames: &mut CallStack,
        continuation: RequireContinuation,
        loader: Value,
        name: LuaString,
    ) -> VmResult<()> {
        let mut arguments = vec![Value::String(name)];
        if !self.compatibility.is_lua51_family() {
            arguments.push(continuation.loader_data.clone());
        }
        match loader {
            Value::NativeFunction(function) => {
                let results = self.call_native(frames, function, &arguments)?;
                self.finish_require_result(frames, continuation, results.first().cloned())
            }
            Value::Closure(function) => {
                let body = self.closure_body(function)?;
                let frame = self.make_frame(
                    body.chunk(),
                    arguments,
                    body.shared_cells(),
                    Some(ReturnTarget::Require(Box::new(continuation))),
                    None,
                    body.environment(),
                    Some(function),
                )?;
                frames.push_within_limit(frame)?;
                Ok(())
            }
            Value::BuiltinFunction(builtin) => match self.call_builtin(builtin, &arguments) {
                Err(VmError::InvalidDispatch(_)) => self.push_builtin_call_frame(
                    frames,
                    builtin,
                    arguments,
                    ReturnTarget::Require(Box::new(continuation)),
                ),
                results => {
                    self.finish_require_result(frames, continuation, results?.first().cloned())
                }
            },
            value => Err(RuntimeError::TypeError {
                operation: "call module loader",
                expected: "function",
                found: value.type_name(),
            }
            .into()),
        }
    }

    pub(crate) fn finish_require_result(
        &mut self,
        frames: &mut CallStack,
        continuation: RequireContinuation,
        result: Option<Value>,
    ) -> VmResult<()> {
        self.require_sentinels
            .remove(&(continuation.loaded, continuation.name.clone()));
        let key = Value::String(continuation.name.clone());
        let value = match result {
            Some(Value::Nil) | None => {
                let value = self.runtime.table_get(continuation.loaded, &key)?;
                if matches!(value, Value::Nil) || value == Value::Table(continuation.sentinel) {
                    self.runtime
                        .table_set(continuation.loaded, &key, Value::Boolean(true))?;
                    Value::Boolean(true)
                } else {
                    value
                }
            }
            Some(value) => {
                self.runtime
                    .table_set(continuation.loaded, &key, value.clone())?;
                value
            }
        };
        let mut results = vec![value];
        if matches!(
            self.compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ) {
            results.push(continuation.loader_data);
        }
        self.finish_require(frames, continuation.completion, results)
    }

    pub(crate) fn finish_require(
        &mut self,
        frames: &mut CallStack,
        completion: RequireCompletion,
        results: Vec<Value>,
    ) -> VmResult<()> {
        let mut frame = frames.last_mut().expect("require caller frame exists");
        match completion {
            RequireCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                results,
            )?,
            RequireCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    results,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn is_require_sentinel(
        &self,
        loaded: TableHandle,
        name: &LuaString,
        value: &Value,
    ) -> bool {
        self.require_sentinels
            .get(&(loaded, name.clone()))
            .is_some_and(|sentinel| *value == Value::Table(*sentinel))
    }

    pub(crate) fn resume_load_pipeline(
        &self,
        pipeline: LoadPipeline,
        input: LoadInput,
    ) -> PipelinePoll {
        match pipeline {
            LoadPipeline::Undecided {
                source_name,
                mut bytes,
            } => match input {
                LoadInput::Bytes(piece) => {
                    bytes.extend_from_slice(piece.as_bytes());
                    if binary::claims(&bytes) {
                        PipelinePoll::NeedInput(LoadPipeline::Binary(bytes))
                    } else if BINARY_MAGIC.starts_with(&bytes) {
                        PipelinePoll::NeedInput(LoadPipeline::Undecided { source_name, bytes })
                    } else {
                        self.start_text_load(source_name, bytes)
                    }
                }
                LoadInput::End => self.start_text_then_end(source_name, bytes),
            },
            LoadPipeline::Text(session) => map_load_poll(session.resume(input)),
            LoadPipeline::Binary(mut bytes) => match input {
                LoadInput::Bytes(piece) => {
                    bytes.extend_from_slice(piece.as_bytes());
                    PipelinePoll::NeedInput(LoadPipeline::Binary(bytes))
                }
                LoadInput::End => match binary::load(&bytes) {
                    Ok(chunk) => PipelinePoll::Loaded(chunk),
                    Err(error) => PipelinePoll::Failed(LuaString::from(error.to_string())),
                },
            },
        }
    }

    pub(crate) fn start_text_load(&self, source_name: LuaString, bytes: Vec<u8>) -> PipelinePoll {
        let Some(loader) = self.chunk_loader.as_deref() else {
            return PipelinePoll::Failed(LuaString::from("no source loader configured"));
        };
        map_load_poll(
            loader
                .begin(source_name)
                .resume(LoadInput::Bytes(LuaString::from(bytes))),
        )
    }

    pub(crate) fn start_text_then_end(
        &self,
        source_name: LuaString,
        bytes: Vec<u8>,
    ) -> PipelinePoll {
        let Some(loader) = self.chunk_loader.as_deref() else {
            return PipelinePoll::Failed(LuaString::from("no source loader configured"));
        };
        let session = loader.begin(source_name);
        let poll = if bytes.is_empty() {
            LoadPoll::NeedInput(session)
        } else {
            session.resume(LoadInput::Bytes(LuaString::from(bytes)))
        };
        match poll {
            LoadPoll::NeedInput(session) => map_load_poll(session.resume(LoadInput::End)),
            LoadPoll::Loaded(chunk) => PipelinePoll::Loaded(chunk),
            LoadPoll::Failed(error) => PipelinePoll::Failed(error),
        }
    }

    pub(crate) fn finish_load_pipeline(&mut self, poll: PipelinePoll) -> Vec<Value> {
        match poll {
            PipelinePoll::Loaded(chunk) => match chunk.validate() {
                Ok(()) => match self.install_loaded_closure(chunk) {
                    Ok(handle) => vec![Value::Closure(handle)],
                    Err(error) => vec![Value::Nil, Value::from(error.to_string())],
                },
                Err(error) => vec![Value::Nil, Value::from(error.to_string())],
            },
            PipelinePoll::Failed(error) => vec![Value::Nil, Value::String(error)],
            PipelinePoll::NeedInput(_) => vec![
                Value::Nil,
                Value::from("loader requested input after end of source"),
            ],
        }
    }

    pub(crate) fn install_loaded_closure(&mut self, chunk: Chunk) -> VmResult<ClosureHandle> {
        let cells = self.initial_chunk_upvalues(&chunk, self.thread_environment)?;
        self.install_closure(Arc::new(chunk), cells, self.thread_environment)
    }

    pub(crate) fn initial_chunk_upvalues(
        &mut self,
        chunk: &Chunk,
        environment: TableHandle,
    ) -> VmResult<Vec<CellHandle>> {
        let mut cells = Vec::with_capacity(chunk.upvalues.len());
        for descriptor in &chunk.upvalues {
            if !matches!(descriptor.source, crate::bytecode::UpvalueSource::Detached) {
                return Err(VmError::UnsupportedUpvalues);
            }
            let value = if matches!(
                self.compatibility,
                CompatibilityProfile::Lua52
                    | CompatibilityProfile::Lua53
                    | CompatibilityProfile::Lua54
                    | CompatibilityProfile::Lua55
            ) && descriptor.name.as_bytes() == b"_ENV"
            {
                Value::Table(environment)
            } else {
                Value::Nil
            };
            cells.push(self.runtime.create_cell(value)?);
        }
        Ok(cells)
    }

    pub(crate) fn start_loadfile(
        &mut self,
        frames: &mut CallStack,
        completion: LoadCompletion,
        arguments: &[Value],
    ) -> VmResult<()> {
        let results = match arguments.first() {
            Some(Value::String(path)) => {
                let results = self.load_file(path, arguments.get(1));
                let environment = if self.compatibility.is_lua51_family() {
                    None
                } else {
                    arguments.get(2)
                };
                self.apply_load_environment(results, environment)
            }
            Some(Value::Nil) | None => vec![Value::Nil, Value::from("loadfile expects a string")],
            Some(value) => {
                return Err(RuntimeError::bad_argument(1, "string", Some(value)).into());
            }
        };
        self.finish_load(frames, completion, results)
    }

    /// Implements Lua 5.1's `dofile`: compile a named file (or standard
    /// input when its argument is absent or nil), then invoke the produced
    /// chunk without converting loader failures into return values.
    pub(crate) fn start_dofile(
        &mut self,
        frames: &mut CallStack,
        completion: DofileCompletion,
        arguments: &[Value],
    ) -> VmResult<()> {
        let results = match arguments.first() {
            Some(Value::String(path)) => self.load_file(path, None),
            Some(Value::Nil) | None => self.load_stdin_file(),
            Some(value) => {
                return Err(RuntimeError::bad_argument(1, "string", Some(value)).into());
            }
        };
        let [Value::Closure(function)] = results.as_slice() else {
            let error = results
                .get(1)
                .cloned()
                .unwrap_or_else(|| Value::from("dofile loader returned an invalid result"));
            return Err(raised_value(error));
        };
        if frames.len() >= MAX_LUA_CALL_FRAMES {
            return Err(RuntimeError::message("stack overflow").into());
        }
        let body = self.closure_body(*function)?;
        let frame = self.make_frame(
            body.chunk(),
            Vec::new(),
            body.shared_cells(),
            Some(ReturnTarget::Dofile(completion)),
            None,
            body.environment(),
            Some(*function),
        )?;
        frames.push(frame);
        Ok(())
    }

    pub(crate) fn load_file(&mut self, path: &LuaString, mode: Option<&Value>) -> Vec<Value> {
        let display = String::from_utf8_lossy(path.as_bytes());
        let mut source_name = Vec::with_capacity(path.len() + 1);
        source_name.push(b'@');
        source_name.extend_from_slice(path.as_bytes());
        let result = self
            .open_host_source(path.as_bytes())
            .and_then(|reader| self.load_file_source(reader, &LuaString::from(source_name), mode));
        result.unwrap_or_else(|error| {
            vec![
                Value::Nil,
                Value::from(self.file_load_error(&display, &error)),
            ]
        })
    }

    fn load_file_source(
        &mut self,
        source: HostSource,
        source_name: &LuaString,
        mode: Option<&Value>,
    ) -> std::io::Result<Vec<Value>> {
        let mode = if self.compatibility.is_lua51_family() {
            None
        } else {
            match mode {
                None | Some(Value::Nil) => None,
                Some(Value::String(mode)) => Some(mode.as_bytes()),
                Some(value) => {
                    return Ok(vec![
                        Value::Nil,
                        Value::from(format!(
                            "load mode must be a string, got {}",
                            value.type_name()
                        )),
                    ]);
                }
            }
        };
        let loader = self.chunk_loader.as_deref().unwrap_or(&MissingSourceLoader);
        let poll = match source.policy {
            FileLoadPolicy::Pipelined => {
                load_file_reader_pipelined(loader, source_name, source.reader, mode)?
            }
            FileLoadPolicy::Complete => {
                load_small_file_reader(loader, source_name, source.reader, mode)?
            }
            FileLoadPolicy::Streaming => {
                load_file_reader(loader, source_name, source.reader, mode)?
            }
        };
        Ok(self.finish_load_pipeline(map_load_poll(poll)))
    }

    /// `luaL_loadfilex`'s message for a file it could not open or read. C
    /// opens a directory fine and fails reading it; Lua 5.5 leaves out the
    /// reason for a read failure.
    pub(crate) fn file_load_error(&self, name: &str, error: &std::io::Error) -> String {
        let reason = crate::runtime::io_error_text(error);
        // EISDIR
        if crate::runtime::io_error_code(error) == Some(21) {
            return if self.compatibility == CompatibilityProfile::Lua55 {
                format!("cannot read {name}")
            } else {
                format!("cannot read {name}: {reason}")
            };
        }
        format!("cannot open {name}: {reason}")
    }

    pub(crate) fn validate_load_mode(
        &self,
        mode: Option<&Value>,
        source: &[u8],
    ) -> Result<(), String> {
        if self.compatibility.is_lua51_family() {
            return Ok(());
        }
        let mode = match mode {
            None | Some(Value::Nil) => b"bt".as_slice(),
            Some(Value::String(mode)) => mode.as_bytes(),
            Some(value) => {
                return Err(format!(
                    "load mode must be a string, got {}",
                    value.type_name()
                ));
            }
        };
        let binary = binary::claims(source);
        if binary && !mode.contains(&b'b') {
            return Err("attempt to load a binary chunk (mode is 't')".to_owned());
        }
        if !binary && !mode.contains(&b't') {
            return Err("attempt to load a text chunk (mode is 'b')".to_owned());
        }
        Ok(())
    }

    pub(crate) fn load_stdin_file(&mut self) -> Vec<Value> {
        if self.source_reader.is_some() {
            let result = self.read_host_source(None).and_then(|source| {
                self.load_file_source(
                    HostSource::complete(source),
                    &LuaString::from("=stdin"),
                    None,
                )
            });
            return result.unwrap_or_else(|error| vec![Value::Nil, Value::from(error.to_string())]);
        }
        match self.runtime.create_stdin_file() {
            Ok(_) => {}
            Err(error) => return vec![Value::Nil, Value::from(error.to_string())],
        }
        let loader = self.chunk_loader.as_deref().unwrap_or(&MissingSourceLoader);
        // Runtime io.read peeks in Rust's shared stdin buffer. Lock that same
        // buffer so pending lookahead remains visible and I/O errors retain
        // their kinds, including Interrupted for the loader's retry loop.
        let input = std::io::stdin();
        let reader = input.lock();
        match load_file_reader(loader, &LuaString::from("=stdin"), reader, None) {
            Ok(poll) => self.finish_load_pipeline(map_load_poll(poll)),
            Err(error) => vec![Value::Nil, Value::from(error.to_string())],
        }
    }

    fn open_host_source(&self, path: &[u8]) -> std::io::Result<HostSource> {
        if let Some(reader) = &self.source_reader {
            return reader(Some(path)).map(HostSource::complete);
        }
        let path = host_path(path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "path is not valid UTF-8")
        })?;
        let file = std::fs::File::open(path)?;
        let policy = match file.metadata() {
            Ok(metadata) if metadata.is_file() && metadata.len() > 8192 => {
                FileLoadPolicy::Pipelined
            }
            Ok(metadata) if metadata.is_file() => FileLoadPolicy::Complete,
            _ => FileLoadPolicy::Streaming,
        };
        Ok(HostSource {
            reader: Box::new(file),
            policy,
        })
    }

    fn read_host_source(&self, path: Option<&[u8]>) -> std::io::Result<Vec<u8>> {
        self.source_reader
            .as_ref()
            .expect("complete reads require the host source hook")(path)
    }

    /// Lua 5.1 keeps `load` (a reader function) and `loadstring` (a string)
    /// apart; every later profile and `LuaJIT` take either from `load`.
    pub(crate) fn check_load_argument(
        &self,
        builtin: BuiltinFunction,
        arguments: &[Value],
    ) -> VmResult<()> {
        if self.compatibility != CompatibilityProfile::Lua51 {
            return Ok(());
        }
        let argument = arguments.first();
        let accepted = match builtin {
            BuiltinFunction::Load => matches!(
                argument,
                Some(Value::NativeFunction(_) | Value::Closure(_) | Value::BuiltinFunction(_))
            ),
            BuiltinFunction::LoadString => matches!(
                argument,
                Some(Value::String(_) | Value::Integer(_) | Value::Number(_))
            ),
            _ => true,
        };
        if accepted {
            return Ok(());
        }
        let expected = if builtin == BuiltinFunction::LoadString {
            "string"
        } else {
            "function"
        };
        Err(RuntimeError::bad_argument(1, expected, argument).into())
    }

    pub(crate) fn start_load(
        &mut self,
        frames: &mut CallStack,
        completion: LoadCompletion,
        arguments: &[Value],
    ) -> VmResult<()> {
        // Every profile reads a number as its string. Lua 5.1 lets only
        // `loadstring` through (`check_load_argument` stops `load`).
        if let Some(number @ (Value::Integer(_) | Value::Number(_))) = arguments.first() {
            let mut arguments = arguments.to_vec();
            arguments[0] = Value::String(self.runtime.format_value(number)?);
            return self.start_load(frames, completion, &arguments);
        }
        match arguments.first() {
            Some(Value::String(source)) => {
                if let Err(error) = self.validate_load_mode(arguments.get(2), source.as_bytes()) {
                    return self.finish_load(
                        frames,
                        completion,
                        vec![Value::Nil, Value::from(error)],
                    );
                }
                let numeric_name = match arguments.get(1) {
                    Some(number @ (Value::Integer(_) | Value::Number(_))) => {
                        Some(self.runtime.format_value(number)?)
                    }
                    _ => None,
                };
                let source_name = match arguments.get(1) {
                    _ if numeric_name.is_some() => numeric_name.as_ref().unwrap_or(source),
                    Some(Value::String(name)) => name,
                    Some(Value::Nil) | None => source,
                    Some(value) => {
                        return self.finish_load(
                            frames,
                            completion,
                            vec![
                                Value::Nil,
                                Value::from(format!(
                                    "load chunk name must be a string, got {}",
                                    value.type_name()
                                )),
                            ],
                        );
                    }
                };
                let loaded = self.load_string(source, source_name);
                let environment = if self.compatibility.is_lua51_family() {
                    None
                } else {
                    arguments.get(3)
                };
                let results = self.apply_load_environment(loaded, environment);
                self.finish_load(frames, completion, results)
            }
            Some(
                reader @ (Value::NativeFunction(_) | Value::Closure(_) | Value::BuiltinFunction(_)),
            ) => {
                let source_name = match arguments.get(1) {
                    Some(Value::String(name)) => name.clone(),
                    Some(number @ (Value::Integer(_) | Value::Number(_))) => {
                        self.runtime.format_value(number)?
                    }
                    Some(Value::Nil) | None => LuaString::from("=(load)"),
                    Some(value) => {
                        return self.finish_load(
                            frames,
                            completion,
                            vec![
                                Value::Nil,
                                Value::from(format!(
                                    "load chunk name must be a string, got {}",
                                    value.type_name()
                                )),
                            ],
                        );
                    }
                };
                self.run_load_reader(
                    frames,
                    LoadReaderContinuation {
                        reader: reader.clone(),
                        session: LoadPipeline::new(source_name),
                        completion,
                    },
                    None,
                )
            }
            argument => Err(RuntimeError::bad_argument(1, "function", argument).into()),
        }
    }

    /// Drives native and builtin readers in a loop, yielding only for a Lua closure.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn run_load_reader(
        &mut self,
        frames: &mut CallStack,
        mut continuation: LoadReaderContinuation,
        mut results: Option<Vec<Value>>,
    ) -> VmResult<()> {
        loop {
            if let Some(values) = results.take() {
                let value = values.first().unwrap_or(&Value::Nil);
                let input = match value {
                    Value::Nil => LoadInput::End,
                    Value::String(piece) if piece.as_bytes().is_empty() => LoadInput::End,
                    Value::String(piece) => LoadInput::Bytes(piece.clone()),
                    // Lua reads a number as its string, like any string value.
                    number @ (Value::Integer(_) | Value::Number(_)) => {
                        LoadInput::Bytes(self.runtime.format_value(number)?)
                    }
                    _ => {
                        return self.finish_load(
                            frames,
                            continuation.completion,
                            vec![
                                Value::Nil,
                                Value::from("reader function must return a string"),
                            ],
                        );
                    }
                };
                match self.resume_load_pipeline(continuation.session, input) {
                    PipelinePoll::NeedInput(session) => continuation.session = session,
                    terminal => {
                        let results = self.finish_load_pipeline(terminal);
                        return self.finish_load(frames, continuation.completion, results);
                    }
                }
            }

            match continuation.reader.clone() {
                Value::NativeFunction(function) => {
                    results = match self.call_native(frames, function, &[]) {
                        Ok(values) => Some(values),
                        Err(error) => {
                            let error = self.protected_argument_error(
                                &Value::NativeFunction(function),
                                VmError::Runtime(error),
                            );
                            return self.finish_load(
                                frames,
                                continuation.completion,
                                vec![Value::Nil, Value::from(error.to_string())],
                            );
                        }
                    };
                }
                Value::Closure(function) => {
                    let body = match self.closure_body(function) {
                        Ok(body) => body,
                        Err(error) => {
                            return self.finish_load(
                                frames,
                                continuation.completion,
                                vec![Value::Nil, Value::from(error.to_string())],
                            );
                        }
                    };
                    let frame = self.make_frame(
                        body.chunk(),
                        Vec::new(),
                        body.shared_cells(),
                        Some(ReturnTarget::LoadReader(Box::new(continuation))),
                        None,
                        body.environment(),
                        Some(function),
                    )?;
                    frames.push_within_limit(frame)?;
                    return Ok(());
                }
                Value::BuiltinFunction(BuiltinFunction::ProtectedCall) => {
                    results = Some(vec![
                        Value::Boolean(false),
                        Value::from("attempt to call a nil value"),
                    ]);
                }
                Value::BuiltinFunction(BuiltinFunction::TableSort) => {
                    return self.finish_load(
                        frames,
                        continuation.completion,
                        vec![Value::Nil, Value::from("call table.sort with a nil value")],
                    );
                }
                Value::BuiltinFunction(builtin) => {
                    results = match self.call_builtin(builtin, &[]) {
                        Ok(values) => Some(values),
                        Err(VmError::InvalidDispatch(_)) => {
                            return self.push_builtin_call_frame(
                                frames,
                                builtin,
                                Vec::new(),
                                ReturnTarget::LoadReader(Box::new(continuation)),
                            );
                        }
                        Err(error) => {
                            let error = self
                                .protected_argument_error(&Value::BuiltinFunction(builtin), error);
                            return self.finish_load(
                                frames,
                                continuation.completion,
                                vec![Value::Nil, Value::from(error.to_string())],
                            );
                        }
                    };
                }
                value => {
                    return self.finish_load(
                        frames,
                        continuation.completion,
                        vec![
                            Value::Nil,
                            Value::from(format!(
                                "load expects a string or function, got {}",
                                value.type_name()
                            )),
                        ],
                    );
                }
            }
        }
    }

    pub(crate) fn finish_load(
        &mut self,
        frames: &mut CallStack,
        completion: LoadCompletion,
        results: Vec<Value>,
    ) -> VmResult<()> {
        let mut frame = frames.last_mut().expect("load caller frame exists");
        match completion {
            LoadCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                results,
            )?,
            LoadCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    results,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish_dofile(
        &mut self,
        frames: &mut CallStack,
        completion: DofileCompletion,
        results: Vec<Value>,
    ) -> VmResult<()> {
        let mut frame = frames.last_mut().expect("dofile caller frame exists");
        match completion {
            DofileCompletion::Direct(target) => write_call_results(
                &mut frame,
                Instruction::call(target.base, 0, target.results),
                results,
            )?,
            DofileCompletion::Protected(target) => {
                self.write_protected_results(
                    &mut frame,
                    target.base,
                    target.results,
                    true,
                    results,
                )?;
            }
        }
        Ok(())
    }
}
