use super::*;
use crate::vm::stack::CallStack;

#[test]
fn require_calls_host_inserted_package_loader() {
    #[allow(clippy::unnecessary_wraps)]
    fn module(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::Number(84.0)])
    }
    fn searcher(runtime: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        assert_eq!(arguments, &[Value::from("vim.virtual")]);
        Ok(vec![runtime.get_global("test_module_loader")])
    }

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let module = vm.runtime_mut().register_native(module).unwrap();
    let searcher = vm.runtime_mut().register_native(searcher).unwrap();
    vm.runtime_mut()
        .set_global("test_module_loader", Value::NativeFunction(module));
    let Value::Table(package) = vm.runtime().get_global("package") else {
        panic!("package is a table");
    };
    let Value::Table(loaders) = vm
        .runtime()
        .table_get(package, &Value::from("loaders"))
        .unwrap()
    else {
        panic!("package.loaders is a table");
    };
    vm.runtime_mut()
        .table_set(
            loaders,
            &Value::Number(2.0),
            Value::NativeFunction(searcher),
        )
        .unwrap();
    vm.set_source_reader(Box::new(|_| {
        panic!("host searcher must run before path lookup")
    }));
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("vim.virtual")))
            .unwrap(),
        vec![Value::Number(84.0)]
    );
}

#[test]
fn require_resumes_after_lua_package_searcher() {
    #[allow(clippy::unnecessary_wraps)]
    fn module(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::Number(85.0)])
    }

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let module = vm.runtime_mut().register_native(module).unwrap();
    vm.runtime_mut()
        .set_global("test_module_loader", Value::NativeFunction(module));
    let searcher = Chunk::new(
        1,
        vec![Constant::String("test_module_loader".into())],
        vec![Instruction::get_global(0, 0), Instruction::return_(0)],
    );
    let searcher = vm
        .install_closure(searcher, Vec::new(), vm.runtime().global_environment())
        .unwrap();
    let Value::Table(package) = vm.runtime().get_global("package") else {
        panic!("package is a table");
    };
    let Value::Table(loaders) = vm
        .runtime()
        .table_get(package, &Value::from("loaders"))
        .unwrap()
    else {
        panic!("package.loaders is a table");
    };
    vm.runtime_mut()
        .table_set(loaders, &Value::Number(2.0), Value::Closure(searcher))
        .unwrap();
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("vim.lua_virtual")))
            .unwrap(),
        vec![Value::Number(85.0)]
    );
}

#[test]
fn pcall_require_catches_missing_module_after_package_searchers() {
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let searcher = Chunk::new(
        1,
        vec![Constant::String("\n\tno custom module".into())],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let searcher = vm
        .install_closure(searcher, Vec::new(), vm.runtime().global_environment())
        .unwrap();
    let Value::Table(package) = vm.runtime().get_global("package") else {
        panic!("package is a table");
    };
    let Value::Table(loaders) = vm
        .runtime()
        .table_get(package, &Value::from("loaders"))
        .unwrap()
    else {
        panic!("package.loaders is a table");
    };
    vm.runtime_mut()
        .table_set(loaders, &Value::Number(2.0), Value::Closure(searcher))
        .unwrap();
    vm.runtime_mut()
        .table_set(loaders, &Value::Number(3.0), Value::Nil)
        .unwrap();
    let caller = Chunk::new(
        3,
        vec![
            Constant::String("pcall".into()),
            Constant::String("require".into()),
            Constant::String("string.buffer".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert!(matches!(
        vm.execute(&caller).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(error)]
            if error.to_lossy().contains("string.buffer")
    ));
}

#[test]
fn source_reader_supplies_virtual_files_modules_and_stdin() {
    fn loader(source: &LuaString, _: &LuaString) -> Result<Chunk, String> {
        let value = match source.as_bytes() {
            b"virtual file" => 11.0,
            b"virtual module" => 22.0,
            b"virtual stdin" => 33.0,
            _ => return Err("unexpected source".to_owned()),
        };
        Ok(Chunk::new(
            1,
            vec![Constant::Number(value)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        ))
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_reader = Arc::clone(&seen);
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, loader);
    vm.set_source_reader(Box::new(move |path| {
        seen_reader.lock().unwrap().push(path.map(<[u8]>::to_vec));
        match path {
            Some(b"virtual.lua") => Ok(b"virtual file".to_vec()),
            // `require` joins the module path with the platform's separator.
            Some(b"mods/a/b.lua" | b"mods/a\\b.lua") => Ok(b"virtual module".to_vec()),
            None => Ok(b"virtual stdin".to_vec()),
            _ => Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing")),
        }
    }));
    let Value::Table(package) = vm.runtime().get_global("package") else {
        panic!("package is a table");
    };
    vm.runtime_mut()
        .table_set(package, &Value::from("path"), Value::from("mods/?.lua"))
        .unwrap();

    let file = LuaString::from("virtual.lua");
    let compiled_file = vm.load_file(&file, None);
    let [Value::Closure(file_closure)] = compiled_file.as_slice() else {
        panic!("virtual file compiles");
    };
    assert!(vm.closure_body(*file_closure).is_ok());
    let dofile = Chunk::new(
        2,
        vec![Constant::String("dofile".into()), Constant::String(file)],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_open(0, 0),
        ],
    );
    assert_eq!(vm.execute(&dofile).unwrap(), vec![Value::Number(11.0)]);
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("a.b"))).unwrap(),
        vec![Value::Number(22.0)]
    );
    assert!(matches!(
        vm.load_stdin_file().as_slice(),
        [Value::Closure(_)]
    ));
    assert_eq!(
        seen.lock().unwrap().as_slice(),
        [
            Some(b"virtual.lua".to_vec()),
            Some(b"virtual.lua".to_vec()),
            Some(format!("mods/a{}b.lua", std::path::MAIN_SEPARATOR).into_bytes()),
            None,
        ]
    );
}

#[test]
fn source_reader_failure_does_not_fall_back_to_host_file() {
    let path = std::env::temp_dir().join(format!("oximond-source-reader-{}", std::process::id()));
    fs::write(&path, b"host source").unwrap();
    let mut vm = Vm::new();
    vm.set_source_reader(Box::new(|_| {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ))
    }));
    let result = vm.load_file(&LuaString::from(path.to_string_lossy().into_owned()), None);
    assert!(
        matches!(result.as_slice(), [Value::Nil, Value::String(message)] if message.to_lossy().contains("denied"))
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn module_resolver_runs_after_preload_and_before_path_search() {
    fn loader(source: &LuaString, source_name: &LuaString) -> Result<Chunk, String> {
        assert_eq!(source.as_bytes(), b"resolver source");
        assert_eq!(source_name.as_bytes(), b"=virtual-module");
        Ok(Chunk::new(
            1,
            vec![Constant::Number(72.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        ))
    }
    let calls = Arc::new(Mutex::new(Vec::new()));
    let resolver_calls = Arc::clone(&calls);
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, loader);
    vm.set_module_resolver(Box::new(move |name| {
        resolver_calls.lock().unwrap().push(name.to_vec());
        Ok(Some((
            b"resolver source".to_vec(),
            b"=virtual-module".to_vec(),
        )))
    }));
    vm.set_source_reader(Box::new(|_| {
        panic!("resolved module must not search files")
    }));
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("virtual")))
            .unwrap(),
        vec![Value::Number(72.0)]
    );
    assert_eq!(calls.lock().unwrap().as_slice(), [b"virtual".to_vec()]);
}

#[test]
fn require_caches_preloads_and_uses_true_for_nil_module_results() {
    fn table_module(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![runtime.create_table_value()?])
    }
    #[allow(clippy::unnecessary_wraps)]
    fn nil_module(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(Vec::new())
    }
    #[allow(clippy::unnecessary_wraps)]
    fn number_module(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::Number(17.0)])
    }

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let (_, preload) = package_tables(vm.runtime());
    for (name, loader) in [
        ("table", table_module as crate::runtime::NativeCallback),
        ("nil", nil_module as crate::runtime::NativeCallback),
        ("number", number_module as crate::runtime::NativeCallback),
    ] {
        let loader = vm.runtime_mut().register_native(loader).unwrap();
        vm.runtime_mut()
            .table_set(preload, &Value::from(name), Value::NativeFunction(loader))
            .unwrap();
    }

    let first = vm
        .execute(&require_chunk(LuaString::from("table")))
        .unwrap();
    let second = vm
        .execute(&require_chunk(LuaString::from("table")))
        .unwrap();
    assert!(Runtime::raw_equal(&first[0], &second[0]));
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("nil"))).unwrap(),
        vec![Value::Boolean(true)]
    );
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("number")))
            .unwrap(),
        vec![Value::Number(17.0)]
    );
}

#[test]
fn require_keeps_its_package_state_when_the_global_binding_is_cleared() {
    #[allow(clippy::unnecessary_wraps)] // Native test callbacks use the runtime's fallible ABI.
    fn module(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::Number(51.0)])
    }

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let (_, preload) = package_tables(vm.runtime());
    let loader = vm.runtime_mut().register_native(module).unwrap();
    vm.runtime_mut()
        .table_set(
            preload,
            &Value::from("retained"),
            Value::NativeFunction(loader),
        )
        .unwrap();

    vm.runtime_mut().set_global("package", Value::Nil);
    let roots = Vm::gc_roots(vm.thread_environment, &mut CallStack::new(), &[]);
    vm.runtime_mut().collect_full(roots);

    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("retained")))
            .unwrap(),
        vec![Value::Number(51.0)]
    );
    assert_eq!(vm.runtime().get_global("package"), Value::Nil);
}

#[test]
fn require_accepts_a_lua51_module_that_replaces_its_loading_sentinel() {
    let loader = Chunk::with_parts(
        2,
        1,
        vec![
            Constant::String("module".into()),
            Constant::String("_M".into()),
        ],
        vec![
            Instruction::move_(1, 0),
            Instruction::get_global(0, 0),
            Instruction::call(0, 1, 0),
            Instruction::get_global(0, 1),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let closure = vm
        .install_closure(loader, Vec::new(), vm.runtime().global_environment())
        .unwrap();
    let (loaded_table, preload) = package_tables(vm.runtime());
    vm.runtime_mut()
        .table_set(preload, &Value::from("P1"), Value::Closure(closure))
        .unwrap();

    let result = vm.execute(&require_chunk(LuaString::from("P1"))).unwrap();
    let [Value::Table(module)] = result.as_slice() else {
        panic!("module returns its package table: {result:?}");
    };
    assert_eq!(vm.runtime().get_global("P1"), Value::Table(*module));
    assert_eq!(
        vm.runtime()
            .table_get(loaded_table, &Value::from("P1"))
            .unwrap(),
        Value::Table(*module)
    );
    assert_eq!(
        vm.runtime()
            .table_get(*module, &Value::from("_NAME"))
            .unwrap(),
        Value::from("P1")
    );
}

#[test]
fn lua51_module_handles_nil_non_table_and_table_loaded_entries() {
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua51);
    install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua51).unwrap();
    let (loaded, _) = package_tables(vm.runtime());

    // A nil cache entry creates both the dotted global table and the
    // corresponding package.loaded entry.
    vm.execute(&module_chunk(LuaString::from("created")))
        .unwrap();
    let Value::Table(created) = vm.runtime().get_global("created") else {
        panic!("module created a global table");
    };
    assert_eq!(
        vm.runtime()
            .table_get(loaded, &Value::from("created"))
            .unwrap(),
        Value::Table(created)
    );

    // Any non-table cache value is replaced by the existing global module
    // table, as luaL_findtable does in the Lua 5.1 implementation.
    let reused = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut().set_global("reused", Value::Table(reused));
    vm.runtime_mut()
        .table_set(loaded, &Value::from("reused"), Value::Number(1.0))
        .unwrap();
    vm.execute(&module_chunk(LuaString::from("reused")))
        .unwrap();
    assert_eq!(
        vm.runtime()
            .table_get(loaded, &Value::from("reused"))
            .unwrap(),
        Value::Table(reused)
    );

    // A regular table cache entry wins, even if the same global name is a
    // different table. The require-specific sentinel case is exercised by
    // require_accepts_a_lua51_module_that_replaces_its_loading_sentinel.
    let cached = vm.runtime_mut().create_table().unwrap();
    let unrelated_global = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(cached, &Value::from("_NAME"), Value::from("prior.name"))
        .unwrap();
    vm.runtime_mut()
        .table_set(loaded, &Value::from("cached"), Value::Table(cached))
        .unwrap();
    vm.runtime_mut()
        .set_global("cached", Value::Table(unrelated_global));
    vm.execute(&module_chunk(LuaString::from("cached")))
        .unwrap();
    assert_eq!(
        vm.runtime()
            .table_get(loaded, &Value::from("cached"))
            .unwrap(),
        Value::Table(cached)
    );
    assert_eq!(
        vm.runtime().get_global("cached"),
        Value::Table(unrelated_global)
    );
    assert_eq!(
        vm.runtime()
            .table_get(cached, &Value::from("_NAME"))
            .unwrap(),
        Value::from("prior.name")
    );
}

#[test]
fn require_loads_dotted_names_as_raw_byte_sources_and_reports_missing_modules() {
    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    fn raw_loader(source: &LuaString, source_name: &LuaString) -> Result<Chunk, String> {
        let module = format!("a{}b.lua", std::path::MAIN_SEPARATOR);
        if source.as_bytes() != b"\xffmodule"
            || !source_name.as_bytes().ends_with(module.as_bytes())
        {
            return Err("unexpected module source".to_owned());
        }
        Ok(Chunk::new(
            1,
            vec![Constant::Number(23.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        ))
    }

    let directory = std::env::temp_dir().join(format!(
        "oximond-require-{}",
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(directory.join("a")).unwrap();
    fs::write(directory.join("a/b.lua"), b"\xffmodule").unwrap();

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, raw_loader);
    let Value::Table(package) = vm.runtime().get_global("package") else {
        panic!("package is a table");
    };
    let path = format!("{}/?.lua", directory.display());
    vm.runtime_mut()
        .table_set(package, &Value::from("path"), Value::from(path))
        .unwrap();
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("a.b"))).unwrap(),
        vec![Value::Number(23.0)]
    );
    assert!(matches!(
        vm.execute(&require_chunk(LuaString::from("missing"))),
        Err(VmError::Runtime(RuntimeError::Message(message)))
            if message.contains("module 'missing' not found") && message.contains("no file")
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn require_leaves_its_sentinel_after_a_recursive_loader_error() {
    let recursive_loader = Chunk::new(
        2,
        vec![
            Constant::String("require".into()),
            Constant::String("loop".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
    );
    let caller = Chunk::new(
        3,
        vec![
            Constant::String("pcall".into()),
            Constant::String("require".into()),
            Constant::String("loop".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let closure = vm
        .install_closure(
            recursive_loader,
            Vec::new(),
            vm.runtime.global_environment(),
        )
        .unwrap();
    let (loaded_table, preload) = package_tables(vm.runtime());
    vm.runtime_mut()
        .table_set(preload, &Value::from("loop"), Value::Closure(closure))
        .unwrap();
    assert!(matches!(
        vm.execute(&caller).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(error)]
            if error.as_bytes().windows(b"loop or previous error".len()).any(|part| part == b"loop or previous error")
    ));
    let sentinel = vm
        .runtime()
        .table_get(loaded_table, &Value::from("loop"))
        .unwrap();
    assert!(matches!(&sentinel, Value::Table(_)));
    assert!(matches!(
        vm.execute(&require_chunk(LuaString::from("loop"))),
        Err(VmError::Runtime(RuntimeError::Message(message)))
            if message.contains("loop or previous error")
    ));
    assert!(Runtime::raw_equal(
        &vm.runtime()
            .table_get(loaded_table, &Value::from("loop"))
            .unwrap(),
        &sentinel
    ));

    // `package.loaded` is public state. Clearing a failed module's marker
    // asks Lua to attempt the loader again, rather than preserving a VM
    // side-channel that makes the module permanently unloadable.
    vm.runtime_mut()
        .table_set(loaded_table, &Value::from("loop"), Value::Nil)
        .unwrap();
    assert!(matches!(
        vm.execute(&caller).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(error)]
            if error.as_bytes().windows(b"loop or previous error".len()).any(|part| part == b"loop or previous error")
    ));
    let retried_sentinel = vm
        .runtime()
        .table_get(loaded_table, &Value::from("loop"))
        .unwrap();
    assert!(matches!(&retried_sentinel, Value::Table(_)));
    assert!(!Runtime::raw_equal(&retried_sentinel, &sentinel));

    // The VM uses a private table for Lua's opaque require marker. A
    // later direct module call must still treat that marker as non-table,
    // replace it, and make future require calls use the module table.
    vm.execute(&module_chunk(LuaString::from("loop"))).unwrap();
    let Value::Table(module) = vm
        .runtime()
        .table_get(loaded_table, &Value::from("loop"))
        .unwrap()
    else {
        panic!("module replaced the sentinel with a table");
    };
    assert!(!Runtime::raw_equal(
        &Value::Table(module),
        &retried_sentinel
    ));
    assert_eq!(vm.runtime().get_global("loop"), Value::Table(module));
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("loop"))).unwrap(),
        vec![Value::Table(module)]
    );
}

#[test]
fn require_does_not_publish_a_sentinel_for_a_source_loader_failure() {
    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    fn failing_loader(_: &LuaString, _: &LuaString) -> Result<Chunk, String> {
        Err("synthetic module compile failure".to_owned())
    }

    let directory = std::env::temp_dir().join(format!(
        "oximond-require-failure-{}",
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("bad.lua"), b"bad source").unwrap();
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, failing_loader);
    let (loaded, _) = package_tables(vm.runtime());
    let Value::Table(package) = vm.runtime().get_global("package") else {
        panic!("package is a table");
    };
    vm.runtime_mut()
        .table_set(
            package,
            &Value::from("path"),
            Value::from(format!("{}/?.lua", directory.display())),
        )
        .unwrap();
    assert!(matches!(
        vm.execute(&require_chunk(LuaString::from("bad"))),
        Err(VmError::Runtime(RuntimeError::Message(message)))
            if message == format!("error loading module 'bad' from file '{}/bad.lua':\n\tsynthetic module compile failure", directory.display())
    ));
    assert_eq!(
        vm.runtime().table_get(loaded, &Value::from("bad")).unwrap(),
        Value::Nil
    );
    assert!(matches!(
        vm.execute(&require_chunk(LuaString::from("bad"))),
        Err(VmError::Runtime(RuntimeError::Message(message)))
            if message == format!("error loading module 'bad' from file '{}/bad.lua':\n\tsynthetic module compile failure", directory.display())
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn require_returns_a_value_that_replaces_the_loading_sentinel() {
    let loader = Chunk::with_varargs(
        3,
        0,
        vec![
            Constant::String("package".into()),
            Constant::String("loaded".into()),
            Constant::Number(25.0),
            Constant::String("require".into()),
            Constant::String("C".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::get_table(0, 0, 1),
            Instruction::vararg(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::set_table(0, 1, 2),
            Instruction::get_global(0, 3),
            Instruction::load_constant(1, 4),
            Instruction::call(0, 1, 1),
            Instruction::return_(0),
        ],
        vec![],
        vec![],
        None,
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    let closure = vm
        .install_closure(loader, Vec::new(), vm.runtime().global_environment())
        .unwrap();
    let (loaded_table, preload) = package_tables(vm.runtime());
    vm.runtime_mut()
        .table_set(preload, &Value::from("C"), Value::Closure(closure))
        .unwrap();

    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("C"))).unwrap(),
        vec![Value::Number(25.0)]
    );
    assert_eq!(
        vm.runtime()
            .table_get(loaded_table, &Value::from("C"))
            .unwrap(),
        Value::Number(25.0)
    );
}

#[test]
fn loadfile_preserves_source_bytes_and_filename_metadata() {
    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);
    fn loader(source: &LuaString, source_name: &LuaString) -> Result<Chunk, String> {
        if source.as_bytes() != b"\xffsource" || !source_name.as_bytes().starts_with(b"@") {
            return Err("unexpected loadfile input".to_owned());
        }
        Ok(Chunk::new(
            1,
            vec![Constant::Number(41.0)],
            vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
        ))
    }
    let path = std::env::temp_dir().join(format!(
        "oximond-loadfile-{}.lua",
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, b"\xffsource").unwrap();
    let program = Chunk::new(
        2,
        vec![
            Constant::String("loadfile".into()),
            Constant::String(LuaString::from(path.to_string_lossy().into_owned())),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, loader);
    assert_eq!(vm.execute(&program).unwrap(), vec![Value::Number(41.0)]);
    fs::remove_file(path).unwrap();

    let missing = Chunk::new(
        2,
        vec![
            Constant::String("loadfile".into()),
            Constant::String("definitely-missing-oximond-file".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert!(matches!(
        vm.execute(&missing).unwrap().as_slice(),
        [Value::Nil, Value::String(message)] if message.as_bytes().starts_with(b"cannot open")
    ));
}

#[test]
fn dofile_executes_a_named_chunk_and_preserves_all_results() {
    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);
    fn loader(source: &LuaString, source_name: &LuaString) -> Result<Chunk, String> {
        if source.as_bytes() != b"dofile source" || !source_name.as_bytes().starts_with(b"@") {
            return Err("unexpected dofile input".to_owned());
        }
        Ok(Chunk::new(
            2,
            vec![Constant::Number(41.0), Constant::String("answer".into())],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::return_values(0, 2),
            ],
        ))
    }

    let path = std::env::temp_dir().join(format!(
        "oximond-dofile-{}.lua",
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, b"dofile source").unwrap();
    let program = Chunk::new(
        2,
        vec![
            Constant::String("dofile".into()),
            Constant::String(LuaString::from(path.to_string_lossy().into_owned())),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::return_open(0, 0),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, loader);
    assert_eq!(
        vm.execute(&program).unwrap(),
        vec![Value::Number(41.0), Value::from("answer")]
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn protected_dofile_preserves_errors_from_the_loaded_chunk() {
    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);
    #[allow(clippy::unnecessary_wraps)]
    fn loader(_: &LuaString, _: &LuaString) -> Result<Chunk, String> {
        Ok(Chunk::new(
            2,
            vec![
                Constant::String("error".into()),
                Constant::String("dofile failed".into()),
            ],
            vec![
                Instruction::get_global(0, 0),
                Instruction::load_constant(1, 1),
                Instruction::call(0, 1, 0),
                Instruction::return_(0),
            ],
        ))
    }

    let path = std::env::temp_dir().join(format!(
        "oximond-dofile-error-{}.lua",
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, b"dofile source").unwrap();
    let program = Chunk::new(
        3,
        vec![
            Constant::String("pcall".into()),
            Constant::String("dofile".into()),
            Constant::String(LuaString::from(path.to_string_lossy().into_owned())),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    set_function_loader(&mut vm, loader);
    assert_eq!(
        vm.execute(&program).unwrap(),
        vec![Value::Boolean(false), Value::from("dofile failed")]
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn loadstring_keeps_leading_shell_comment_bytes() {
    fn observes_unmodified_source(source: &LuaString, _: &LuaString) -> Result<Chunk, String> {
        if source.as_bytes() == b"# x" {
            Err("loadstring kept the leading hash".to_owned())
        } else {
            Err("loadstring was unexpectedly preprocessed".to_owned())
        }
    }
    let mut vm = Vm::new();
    set_function_loader(&mut vm, observes_unmodified_source);
    assert!(matches!(
        vm.load_string(&LuaString::from("# x"), &LuaString::from("=(test)")).as_slice(),
        [Value::Nil, Value::String(message)] if message.as_bytes() == b"loadstring kept the leading hash"
    ));
}

#[test]
fn load_creates_a_validated_closure_without_executing_it() {
    fn loader(source: &LuaString, _: &LuaString) -> Result<Chunk, String> {
        match source.as_bytes() {
            b"answer" => Ok(Chunk::new(
                1,
                vec![Constant::Number(42.0)],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
            )),
            b"invalid" => Ok(Chunk::new(0, vec![], vec![])),
            [0xff] => Ok(Chunk::new(
                1,
                vec![Constant::Number(9.0)],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
            )),
            _ => Err("syntax error".to_owned()),
        }
    }

    let loaded = Chunk::new(
        2,
        vec![
            Constant::String("load".into()),
            Constant::String("answer".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 2),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    set_function_loader(&mut vm, loader);
    assert_eq!(vm.execute(&loaded).unwrap(), vec![Value::Number(42.0)]);

    let nil_chunk_name = Chunk::new(
        3,
        vec![
            Constant::String("load".into()),
            Constant::String("answer".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_nil(2),
            Instruction::call(0, 2, 1),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(
        vm.execute(&nil_chunk_name).unwrap(),
        vec![Value::Number(42.0)]
    );

    let bad_source = Chunk::new(
        2,
        vec![
            Constant::String("load".into()),
            Constant::String("invalid".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert!(matches!(
        vm.execute(&bad_source).unwrap().as_slice(),
        [Value::Nil, Value::String(_)]
    ));

    let byte_source = Chunk::new(
        2,
        vec![
            Constant::String("load".into()),
            Constant::String(LuaString::from(vec![0xff])),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(vm.execute(&byte_source).unwrap(), vec![Value::Number(9.0)]);

    let mut without_loader = Vm::new();
    without_loader
        .runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    assert!(matches!(
        without_loader.execute(&bad_source).unwrap().as_slice(),
        [Value::Nil, Value::String(_)]
    ));
}

#[test]
fn string_dump_and_binary_load_execute_without_a_text_loader() {
    let closure = Chunk::new(
        1,
        vec![Constant::Number(42.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let mut vm = Vm::new();
    let original = vm
        .install_closure(closure, Vec::new(), vm.runtime.global_environment())
        .unwrap();
    let Value::String(bytes) = vm
        .string_dump(&[Value::Closure(original)])
        .unwrap()
        .remove(0)
    else {
        panic!("string.dump returns bytes");
    };
    assert!(binary::claims(bytes.as_bytes()));
    let Value::Closure(loaded) = vm
        .load_string(&bytes, &LuaString::from("ignored"))
        .remove(0)
    else {
        panic!("binary load returns a closure");
    };
    assert_eq!(
        call_closure(&mut vm, Value::Closure(loaded)),
        vec![Value::Number(42.0)]
    );
    assert!(matches!(
        vm.string_dump(&[Value::BuiltinFunction(BuiltinFunction::Next)]),
        Err(VmError::Runtime(RuntimeError::TypeError { .. }))
    ));
}

#[test]
fn load_sessions_preserve_bytes_names_and_direct_string_behavior() {
    let loaded = Chunk::new(
        1,
        vec![Constant::Number(42.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let trace = Arc::new(Mutex::new(LoaderTrace::default()));
    let mut vm = Vm::new();
    vm.set_chunk_loader(Box::new(TracingLoader::new(Arc::clone(&trace), loaded)));
    let source = LuaString::from(b"\xff source" as &[u8]);
    let source_name = LuaString::from(b"=\xfe named" as &[u8]);
    let Value::Closure(closure) = vm.load_string(&source, &source_name).remove(0) else {
        panic!("successful direct load returns a closure");
    };
    assert_eq!(
        call_closure(&mut vm, Value::Closure(closure)),
        vec![Value::Number(42.0)]
    );
    let trace = trace.lock().unwrap();
    assert_eq!(trace.names, vec![source_name]);
    assert_eq!(trace.inputs, vec![LoadInput::Bytes(source), LoadInput::End]);
}

#[test]
fn load_reader_stops_on_a_terminal_session_and_sends_end_once() {
    let loaded = Chunk::new(
        1,
        vec![Constant::Number(42.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let trace = Arc::new(Mutex::new(LoaderTrace::default()));
    let program = Chunk::new(
        2,
        vec![
            Constant::String("load".into()),
            Constant::String("reader".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    vm.set_chunk_loader(Box::new(TracingLoader::new(Arc::clone(&trace), loaded)));
    install_sequence_reader(
        &mut vm,
        &[Value::from("answer"), Value::Nil, Value::from("never")],
    );
    assert_eq!(vm.execute(&program).unwrap(), vec![Value::Number(42.0)]);
    assert_eq!(vm.runtime().get_global("reader_index"), Value::Number(3.0));
    assert_eq!(
        trace.lock().unwrap().inputs,
        vec![LoadInput::Bytes(LuaString::from("answer")), LoadInput::End]
    );
}

#[test]
fn load_reader_does_not_pull_after_a_session_failure_and_pcall_keeps_it_a_value() {
    let loaded = Chunk::new(
        1,
        vec![Constant::Number(42.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let trace = Arc::new(Mutex::new(LoaderTrace::default()));
    let program = Chunk::new(
        3,
        vec![
            Constant::String("pcall".into()),
            Constant::String("load".into()),
            Constant::String("reader".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 3),
            Instruction::return_values(0, 3),
        ],
    );
    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    vm.set_chunk_loader(Box::new(TracingLoader::failing_after_bytes(
        Arc::clone(&trace),
        loaded,
        2,
    )));
    install_sequence_reader(
        &mut vm,
        &[Value::from("one"), Value::from("two"), Value::from("never")],
    );
    assert!(matches!(
        vm.execute(&program).unwrap().as_slice(),
        [Value::Boolean(true), Value::Nil, Value::String(message)]
            if message.as_bytes() == b"failed after bytes"
    ));
    assert_eq!(vm.runtime().get_global("reader_index"), Value::Number(3.0));
    assert_eq!(
        trace.lock().unwrap().inputs,
        vec![
            LoadInput::Bytes(LuaString::from("one")),
            LoadInput::Bytes(LuaString::from("two")),
        ]
    );
}

#[test]
fn load_reports_a_protocol_error_when_end_requests_more_input() {
    let trace = Arc::new(Mutex::new(LoaderTrace::default()));
    let mut vm = Vm::new();
    vm.set_chunk_loader(Box::new(TracingLoader::needing_input_after_end(
        trace,
        Chunk::new(1, vec![], vec![Instruction::halt()]),
    )));
    assert!(matches!(
        vm.load_string(&LuaString::from("source"), &LuaString::from("=source"))
            .as_slice(),
        [Value::Nil, Value::String(error)]
            if error.as_bytes() == b"loader requested input after end of source"
    ));
}

#[test]
fn collection_roots_closures_held_only_by_a_load_reader_continuation() {
    let closure = Chunk::new(
        1,
        vec![Constant::Number(23.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let mut vm = Vm::new();
    let handle = vm
        .install_closure(closure, Vec::new(), vm.runtime.global_environment())
        .unwrap();
    let frame = Frame::new(
        Arc::new(Chunk::new(1, vec![], vec![Instruction::halt()])),
        Vec::new(),
        Vec::new(),
        Some(ReturnTarget::LoadReader(Box::new(LoadReaderContinuation {
            reader: Value::Closure(handle),
            session: LoadPipeline::new(LuaString::from("=(test)")),
            completion: LoadCompletion::direct(Instruction::call(0, 0, 0)),
        }))),
        None,
        vm.runtime().global_environment(),
        None,
    );
    let roots = Vm::gc_roots(
        vm.thread_environment,
        &mut CallStack::from(vec![frame]),
        &[],
    );
    assert_eq!(vm.runtime_mut().collect_full(roots).reclaimed(), 0);
    assert_eq!(
        call_closure(&mut vm, Value::Closure(handle)),
        vec![Value::Number(23.0)]
    );
}

#[test]
fn binary_load_handles_malformed_and_one_byte_reader_input() {
    let closure = Chunk::new(
        1,
        vec![Constant::Number(42.0)],
        vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
    );
    let mut vm = Vm::new();
    let original = vm
        .install_closure(closure, Vec::new(), vm.runtime.global_environment())
        .unwrap();
    let Value::String(bytes) = vm
        .string_dump(&[Value::Closure(original)])
        .unwrap()
        .remove(0)
    else {
        panic!("string.dump returns bytes");
    };
    let mut pieces = bytes
        .as_bytes()
        .iter()
        .map(|byte| Value::bytes(vec![*byte]))
        .collect::<Vec<_>>();
    pieces.push(Value::from(""));
    install_sequence_reader(&mut vm, &pieces);
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    let reader_load = Chunk::new(
        2,
        vec![
            Constant::String("load".into()),
            Constant::String("reader".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 1),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(vm.execute(&reader_load).unwrap(), vec![Value::Number(42.0)]);
    let malformed = LuaString::from(b"\x1bOXIMONDbad".as_slice());
    assert!(matches!(
        vm.load_string(&malformed, &LuaString::from("ignored"))
            .as_slice(),
        [Value::Nil, Value::String(_)]
    ));
}

#[test]
fn load_reader_accepts_an_empty_closure_reader() {
    let empty_reader = Chunk::new(
        1,
        vec![],
        vec![Instruction::load_nil(0), Instruction::return_(0)],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("load".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 1),
            Instruction::call(0, 0, 1),
            Instruction::return_(0),
        ],
        vec![empty_reader],
        vec![],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    set_function_loader(&mut vm, reader_loader);
    assert_eq!(vm.execute(&program).unwrap(), vec![Value::Number(0.0)]);
}

#[test]
fn load_reader_preserves_captured_state_across_continuation_reentry() {
    let reader = Chunk::with_parts(
        3,
        0,
        vec![
            Constant::Number(1.0),
            Constant::String("answer".into()),
            Constant::String("".into()),
        ],
        vec![
            Instruction::get_upvalue(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Add, 0, 0, 1),
            Instruction::set_upvalue(0, 0),
            Instruction::load_constant(1, 0),
            Instruction::binary(OpCode::Equal, 2, 0, 1),
            Instruction::jump_if_false(2, 2),
            Instruction::load_constant(0, 1),
            Instruction::return_(0),
            Instruction::load_constant(0, 2),
            Instruction::return_(0),
        ],
        vec![],
        vec![UpvalueDescriptor {
            name: "upvalue".into(),
            source: UpvalueSource::ParentRegister(0),
        }],
    );
    let program = Chunk::with_parts(
        3,
        0,
        vec![Constant::Number(0.0), Constant::String("load".into())],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::get_global(1, 1),
            Instruction::closure(2, 0),
            Instruction::call(1, 1, 1),
            Instruction::call(1, 0, 1),
            Instruction::return_values(0, 2),
        ],
        vec![reader],
        vec![],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    set_function_loader(&mut vm, reader_loader);
    assert_eq!(
        vm.execute(&program).unwrap(),
        vec![Value::Number(2.0), Value::Number(42.0)]
    );
}

#[test]
fn load_reader_converts_failures_and_bad_results_to_load_results() {
    let load_reader = Chunk::new(
        2,
        vec![
            Constant::String("load".into()),
            Constant::String("reader".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    set_function_loader(&mut vm, reader_loader);
    vm.runtime_mut()
        .register_global_native("reader", failing_reader)
        .unwrap();
    assert!(matches!(
        vm.execute(&load_reader).unwrap().as_slice(),
        [Value::Nil, Value::String(message)] if message.as_bytes() == b"reader failed"
    ));

    vm.runtime_mut()
        .register_global_native("reader", boolean_reader)
        .unwrap();
    assert!(matches!(
        vm.execute(&load_reader).unwrap().as_slice(),
        [Value::Nil, Value::String(message)] if message.as_bytes() == b"reader function must return a string"
    ));
}

#[test]
fn load_reader_catches_a_lua_reader_error() {
    let failing_reader = Chunk::new(
        1,
        vec![Constant::String("broken reader".into())],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::unary(OpCode::Negate, 0, 0),
            Instruction::return_(0),
        ],
    );
    let program = Chunk::with_parts(
        2,
        0,
        vec![Constant::String("load".into())],
        vec![
            Instruction::get_global(0, 0),
            Instruction::closure(1, 0),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
        vec![failing_reader],
        vec![],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    set_function_loader(&mut vm, reader_loader);
    assert!(matches!(
        vm.execute(&program).unwrap().as_slice(),
        [Value::Nil, Value::String(message)]
            if message.as_bytes() == b"attempt to perform arithmetic on a string value"
    ));
}

#[test]
fn protected_load_reader_failure_returns_success_and_load_error() {
    let program = Chunk::new(
        3,
        vec![
            Constant::String("pcall".into()),
            Constant::String("load".into()),
            Constant::String("reader".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::get_global(2, 2),
            Instruction::call(0, 2, 3),
            Instruction::return_values(0, 3),
        ],
    );
    let mut vm = Vm::new();
    vm.runtime_mut()
        .set_global("load", Value::BuiltinFunction(BuiltinFunction::Load));
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    vm.runtime_mut()
        .register_global_native("reader", failing_reader)
        .unwrap();
    assert!(matches!(
        vm.execute(&program).unwrap().as_slice(),
        [Value::Boolean(true), Value::Nil, Value::String(message)]
            if message.as_bytes() == b"reader failed"
    ));
}

#[cfg(unix)]
#[test]
fn native_loadfile_delivers_fifo_bytes_before_reading_again() {
    use std::io::Write;
    use std::os::unix::ffi::OsStrExt;
    use std::sync::mpsc;
    use std::time::Duration;

    struct ObservingLoader {
        inner: TracingLoader,
        observed: mpsc::Sender<()>,
    }
    struct ObservingLoad {
        inner: Box<dyn ChunkLoad>,
        observed: Option<mpsc::Sender<()>>,
    }
    impl ChunkLoader for ObservingLoader {
        fn begin(&self, name: LuaString) -> Box<dyn ChunkLoad> {
            Box::new(ObservingLoad {
                inner: self.inner.begin(name),
                observed: Some(self.observed.clone()),
            })
        }
        fn begin_pipelined(&self, _: LuaString) -> Option<Box<dyn ChunkLoad>> {
            panic!("FIFO input must preserve synchronous early failures");
        }
        fn load_complete(&self, _: LuaString, _: LuaString) -> LoadPoll {
            panic!("native files must use the incremental compiler");
        }
    }
    impl ChunkLoad for ObservingLoad {
        fn resume(mut self: Box<Self>, input: LoadInput) -> LoadPoll {
            if matches!(input, LoadInput::Bytes(_))
                && let Some(observed) = self.observed.take()
            {
                observed.send(()).unwrap();
            }
            match self.inner.resume(input) {
                LoadPoll::NeedInput(inner) => {
                    self.inner = inner;
                    LoadPoll::NeedInput(self)
                }
                terminal => terminal,
            }
        }
    }

    let scratch = LoadScratchDirectory::new();
    let directory = &scratch.0;
    let path = directory.join("input.lua");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .unwrap()
            .success()
    );
    let (observed, receive) = mpsc::channel();
    let writer_path = path.clone();
    let writer = std::thread::spawn(move || {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .open(writer_path)
            .unwrap();
        output.write_all(&vec![b'x'; 8192]).unwrap();
        // A whole-file read cannot notify the writer before EOF. The timeout
        // lets the test fail instead of leaving a blocked reader behind.
        let delivered_before_next_read = receive.recv_timeout(Duration::from_secs(5)).is_ok();
        output.write_all(b"tail").unwrap();
        delivered_before_next_read
    });
    let trace = Arc::new(Mutex::new(LoaderTrace::default()));
    let loaded = Chunk::new(1, vec![], vec![Instruction::return_values(0, 0)]);
    let mut vm = Vm::new();
    vm.set_chunk_loader(Box::new(ObservingLoader {
        inner: TracingLoader::new(Arc::clone(&trace), loaded),
        observed,
    }));
    let result = vm.load_file(&LuaString::from(path.as_os_str().as_bytes()), None);
    let delivered_before_next_read = writer.join().unwrap();
    std::fs::remove_file(&path).unwrap();

    assert!(delivered_before_next_read);
    assert!(matches!(result.as_slice(), [Value::Closure(_)]));
    let trace = trace.lock().unwrap();
    assert_eq!(trace.inputs.last(), Some(&LoadInput::End));
    let bytes = trace
        .inputs
        .iter()
        .filter_map(|input| match input {
            LoadInput::Bytes(bytes) => Some(bytes.as_bytes()),
            LoadInput::End => None,
        })
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(bytes, [vec![b'x'; 8192], b"tail".to_vec()].concat());
}

#[test]
fn native_module_searchers_stream_and_preprocess_files() {
    let scratch = LoadScratchDirectory::new();
    let directory = &scratch.0;
    let filename = directory.join("stream.lua");
    let body = vec![b'x'; 20000];
    std::fs::write(
        &filename,
        [b"\xef\xbb\xbf#!/lua\r\n".as_slice(), &body].concat(),
    )
    .unwrap();
    let path = LuaString::from(directory.join("?.lua").to_str().unwrap());
    for legacy in [false, true] {
        let trace = Arc::new(Mutex::new(LoaderTrace::default()));
        let mut vm = Vm::new();
        install_base(vm.runtime_mut()).unwrap();
        vm.set_chunk_loader(Box::new(TracingLoader::new(
            Arc::clone(&trace),
            Chunk::new(
                1,
                vec![Constant::Number(42.0)],
                vec![Instruction::load_constant(0, 0), Instruction::return_(0)],
            ),
        )));
        let Value::Table(package) = vm.runtime().get_global("package") else {
            panic!("package table")
        };
        vm.runtime_mut()
            .table_set(package, &Value::from("path"), Value::String(path.clone()))
            .unwrap();
        if legacy {
            vm.runtime_mut()
                .table_set(package, &Value::from("loaders"), Value::Nil)
                .unwrap();
        }
        assert_eq!(
            vm.execute(&require_chunk(LuaString::from("stream")))
                .unwrap(),
            vec![Value::Number(42.0)]
        );
        let trace = trace.lock().unwrap();
        assert_eq!(trace.inputs.last(), Some(&LoadInput::End));
        assert!(
            trace.inputs.len() >= 4,
            "native module bytes must arrive in bounded fragments"
        );
        let bytes = trace
            .inputs
            .iter()
            .filter_map(|input| match input {
                LoadInput::Bytes(bytes) => Some(bytes.as_bytes()),
                LoadInput::End => None,
            })
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(bytes, [b"\n".as_slice(), &body].concat());
    }
    std::fs::remove_file(filename).unwrap();
}

/// A directory only this test made (the process id and a counter name it,
/// and `create_dir` fails on one that exists); Drop also cleans up failed
/// assertions.
struct LoadScratchDirectory(std::path::PathBuf);

impl LoadScratchDirectory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "oximond-load-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("creating {}: {error}", path.display()),
            }
        }
    }
}

impl Drop for LoadScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[allow(clippy::too_many_lines)] // Verify every native load policy against one shared observing loader.
fn only_large_regular_native_files_select_the_pipeline_factory() {
    use std::cell::Cell;
    use std::rc::Rc;
    struct PipelineLoader {
        inner: TracingLoader,
        begins: Rc<Cell<usize>>,
        complete: Rc<Cell<usize>>,
    }
    impl ChunkLoader for PipelineLoader {
        fn begin(&self, name: LuaString) -> Box<dyn ChunkLoad> {
            self.inner.begin(name)
        }
        fn load_complete(&self, _: LuaString, _: LuaString) -> LoadPoll {
            self.complete.set(self.complete.get() + 1);
            LoadPoll::Loaded(self.inner.chunk.clone())
        }
        fn begin_pipelined(&self, name: LuaString) -> Option<Box<dyn ChunkLoad>> {
            self.begins.set(self.begins.get() + 1);
            Some(self.inner.begin(name))
        }
    }
    let scratch = LoadScratchDirectory::new();
    let filename = scratch.0.join("regular.lua");
    std::fs::write(&filename, vec![b'x'; 8193]).unwrap();
    let begins = Rc::new(Cell::new(0));
    let complete = Rc::new(Cell::new(0));
    let mut vm = Vm::new();
    vm.set_chunk_loader(Box::new(PipelineLoader {
        inner: TracingLoader::new(
            Arc::new(Mutex::new(LoaderTrace::default())),
            Chunk::new(1, vec![], vec![Instruction::return_values(0, 0)]),
        ),
        begins: Rc::clone(&begins),
        complete: Rc::clone(&complete),
    }));
    for (index, size) in [0, 8, 8192].into_iter().enumerate() {
        let small = scratch.0.join(format!("small-{size}.lua"));
        std::fs::write(&small, vec![b'x'; size]).unwrap();
        assert!(matches!(
            vm.load_file(&LuaString::from(small.to_str().unwrap()), None)
                .as_slice(),
            [Value::Closure(_)]
        ));
        assert_eq!(
            complete.get(),
            index + 1,
            "small native files use complete compilation"
        );
        assert_eq!(
            begins.get(),
            0,
            "files fitting in one read stay synchronous"
        );
    }
    let filename = LuaString::from(filename.to_str().unwrap());
    assert!(matches!(
        vm.load_file(&filename, None).as_slice(),
        [Value::Closure(_)]
    ));
    assert_eq!(begins.get(), 1);
    install_base(vm.runtime_mut()).unwrap();
    let Value::Table(package) = vm.runtime().get_global("package") else {
        panic!("package table");
    };
    vm.runtime_mut()
        .table_set(
            package,
            &Value::from("path"),
            Value::String(LuaString::from(scratch.0.join("?.lua").to_str().unwrap())),
        )
        .unwrap();
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("regular")))
            .unwrap(),
        vec![Value::Boolean(true)]
    );
    assert_eq!(begins.get(), 2);
    let Value::Table(loaded) = vm
        .runtime()
        .table_get(package, &Value::from("loaded"))
        .unwrap()
    else {
        panic!("package.loaded table");
    };
    vm.runtime_mut()
        .table_set(loaded, &Value::from("regular"), Value::Nil)
        .unwrap();
    vm.runtime_mut()
        .table_set(package, &Value::from("loaders"), Value::Nil)
        .unwrap();
    assert_eq!(
        vm.execute(&require_chunk(LuaString::from("regular")))
            .unwrap(),
        vec![Value::Boolean(true)]
    );
    assert_eq!(begins.get(), 3);
    vm.set_source_reader(Box::new(|_| Ok(b"return 1".to_vec())));
    assert!(matches!(
        vm.load_file(&filename, None).as_slice(),
        [Value::Closure(_)]
    ));
    assert_eq!(
        begins.get(),
        3,
        "virtual source hooks keep synchronous scheduling"
    );
    assert!(matches!(
        vm.load_stdin_file().as_slice(),
        [Value::Closure(_)]
    ));
    assert_eq!(begins.get(), 3);
}
