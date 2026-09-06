use super::*;

#[test]
fn coroutines_exchange_multiple_values_and_track_status() {
    run("local co; co=coroutine.create(function(first) \
               assert(coroutine.running()==co); \
               local a,b=coroutine.yield(first,2); return a,b,3 \
             end); \
             assert(type(co)=='thread' and coroutine.status(co)=='suspended'); \
             local ok,a,b=coroutine.resume(co,1); \
             assert(ok and a==1 and b==2 and coroutine.status(co)=='suspended'); \
             local ok,x,y,z=coroutine.resume(co,7,8); \
             assert(ok and x==7 and y==8 and z==3 and coroutine.status(co)=='dead'); \
             local ok,message=coroutine.resume(co); \
             assert(not ok and string.find(message,'dead')); \
             assert(coroutine.running()==nil)");
}

#[test]
fn suspended_coroutine_stacks_are_gc_roots_and_have_thread_environments() {
    run("local weak=setmetatable({}, {__mode='v'}); \
             local co=coroutine.create(function() \
               local retained={answer=42}; weak[1]=retained; \
               coroutine.yield(getfenv(0)); \
               assert(retained.answer==42 and weak[1]==retained); \
               return getfenv(0).answer \
             end); \
             local environment={answer=17}; debug.setfenv(co,environment); \
             assert(debug.getfenv(co)==environment); \
             local ok,seen=coroutine.resume(co); assert(ok and seen==environment); \
             collectgarbage(); assert(weak[1]~=nil); \
             local ok,answer=coroutine.resume(co); assert(ok and answer==17)");
}

#[test]
fn profiled_compiler_loader_applies_lua51_parser_rules() {
    let loader = CompilerLoader::with_profile(CompatibilityProfile::Lua51);
    let session = loader.begin(LuaString::from("lua51-reader"));
    let session = match session.resume(LoadInput::Bytes(LuaString::from("a = 1;"))) {
        LoadPoll::NeedInput(session) => session,
        LoadPoll::Loaded(_) | LoadPoll::Failed(_) => panic!("fragment completed early"),
    };
    match session.resume(LoadInput::Bytes(LuaString::from("; "))) {
        LoadPoll::Failed(error) => {
            assert_eq!(
                error.as_bytes(),
                b"[string \"lua51-reader\"]:1: unexpected ';' near ';'"
            );
        }
        LoadPoll::NeedInput(_) | LoadPoll::Loaded(_) => {
            panic!("second semicolon must fail during the reader session")
        }
    }
}

#[test]
fn compiler_loader_matches_named_compile_after_fragmented_input() {
    let loader = CompilerLoader::default();
    let source_name = LuaString::from(b"reader\xff" as &[u8]);
    let expected = compile_named(
        source_name.clone(),
        b"value = 'a\0\xff'",
        CompilerConfig::default(),
    )
    .unwrap();
    let session = loader.begin(source_name);
    let session = match session.resume(LoadInput::Bytes(LuaString::from(b"value = " as &[u8]))) {
        LoadPoll::NeedInput(session) => session,
        LoadPoll::Loaded(_) | LoadPoll::Failed(_) => panic!("fragment must not compile early"),
    };
    let session = match session.resume(LoadInput::Bytes(LuaString::from(b"'a\0\xff'" as &[u8]))) {
        LoadPoll::NeedInput(session) => session,
        LoadPoll::Loaded(_) | LoadPoll::Failed(_) => panic!("fragment must not compile early"),
    };
    match session.resume(LoadInput::End) {
        LoadPoll::Loaded(chunk) => {
            assert_eq!(chunk, expected);
            assert_eq!(chunk.debug.source_name().as_bytes(), b"reader\xff");
        }
        LoadPoll::NeedInput(_) | LoadPoll::Failed(_) => panic!("end must compile exactly once"),
    }
}

#[test]
fn compiler_loader_matches_named_compile_at_every_byte_split() {
    let source = b"function t:call(value, ...) return value, ... end; t:call(1, 2)";
    let source_name = LuaString::from("split-reader");
    let expected = compile_named(source_name.clone(), source, CompilerConfig::default()).unwrap();
    for split in 1..=source.len() {
        let loader = CompilerLoader::default();
        let mut session = loader.begin(source_name.clone());
        for bytes in source.chunks(split) {
            session = match session.resume(LoadInput::Bytes(LuaString::from(bytes))) {
                LoadPoll::NeedInput(session) => session,
                LoadPoll::Loaded(_) | LoadPoll::Failed(_) => {
                    panic!("valid source completed before its explicit end marker")
                }
            };
        }
        match session.resume(LoadInput::End) {
            LoadPoll::Loaded(chunk) => assert_eq!(chunk, expected, "split={split}"),
            LoadPoll::NeedInput(_) | LoadPoll::Failed(_) => {
                panic!("valid split source did not compile at end")
            }
        }
    }
}

#[test]
fn records_lua51_control_condition_and_terminal_lines() {
    // This is the first line-hook case in Lua 5.1's `db.lua`. The `if`
    // keyword is on line 1, but all condition work belongs to line 2.
    // Lua also attributes the synthetic terminal return to the closing
    // `end` on line 7.
    let chunk = compile_named(
        "db.lua",
        b"if\nmath.sin(1)\nthen\n  a=1\nelse\n  a=2\nend\n",
        CompilerConfig::default(),
    )
    .unwrap();
    let call_pc = chunk.debug.call_sites()[0].pc();
    assert!(
        chunk.debug.line_info()[..=call_pc]
            .iter()
            .all(|line| *line == 2)
    );
    assert_eq!(chunk.debug.line_at(chunk.code.len() - 1), Some(7));
}

#[test]
fn dynamic_load_compiles_raw_source_bytes() {
    let mut source = b"local chunk = load([[return '".to_vec();
    source.push(255);
    source.extend_from_slice(b"']]); loaded = chunk()");
    let chunk = compile(&source).unwrap();
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.set_chunk_loader(Box::new(CompilerLoader::default()));
    vm.execute(&chunk).unwrap();
    assert_eq!(
        vm.runtime().get_global("loaded"),
        Value::String(LuaString::from(vec![255]))
    );
}

#[test]
fn lua51_file_methods_support_buffering_numeric_reads_and_lines() {
    run(
        "local name=os.tmpname(); local f=assert(io.open(name,'w+')); \
             assert(f:setvbuf('full',16)); assert(f:write(' 12.5\\nnext')); \
             assert(f:seek('set')==0); assert(f:read('*n')==12.5); \
             assert(f:read()==''); assert(f:read()=='next'); assert(f:close()); \
             assert(io.type(f)=='closed file' and tostring(f)=='file (closed)'); \
             assert(os.remove(name))",
    );
}

#[test]
fn and_or_short_circuit_and_preserve_operands() {
    let chunk = compile(
        "a = false and rhs(); b = true or rhs(); c = nil or 7; d = 3 and 8; \
             less = 1 < 2; equal = 2 >= 2; greater = 3 > 2",
    )
    .unwrap();
    let mut vm = Vm::new();
    vm.runtime_mut()
        .register_global_native("rhs", record_rhs_call)
        .unwrap();
    vm.execute(&chunk).unwrap();

    assert_eq!(vm.runtime().get_global("rhs_was_called"), Value::Nil);
    assert_eq!(vm.runtime().get_global("a"), Value::Boolean(false));
    assert_eq!(vm.runtime().get_global("b"), Value::Boolean(true));
    assert_eq!(vm.runtime().get_global("c"), Value::Number(7.0));
    assert_eq!(vm.runtime().get_global("d"), Value::Number(8.0));
    assert_eq!(vm.runtime().get_global("less"), Value::Boolean(true));
    assert_eq!(vm.runtime().get_global("equal"), Value::Boolean(true));
    assert_eq!(vm.runtime().get_global("greater"), Value::Boolean(true));
}

#[test]
fn compiles_the_pinned_luajit_self_program() {
    let source = r#"
do --- trivial setget
  local t = {}

  function t:set(x)
    self.a=x
  end

  function t:get()
    return self.a
  end

  t:set("foo")
  assert(t:get() == "foo")
  assert(t.a == "foo")

  t:set(42)
  assert(t:get() == 42)
  assert(t.a == 42)
end
"#;
    let chunk = compile(source).unwrap();
    assert_eq!(chunk.prototypes.len(), 2);
    assert_eq!(chunk.prototypes[0].parameter_count, 2);
    assert_eq!(chunk.prototypes[1].parameter_count, 1);
    assert!(
        chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::Closure))
    );
    assert_eq!(chunk.validate(), Ok(()));
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(vm.execute(&chunk).unwrap(), Vec::<Value>::new());
}

#[test]
fn runs_the_pinned_luajit_modulo_program() {
    let source = r"
local assert, floor = assert, math.floor

do --- integer equivalence
  for x=-5,5 do
    for y=-5,5 do
      if y ~= 0 then
        assert(x%y == x-floor(x/y)*y)
      end
    end
  end
end

do --- fractional equivalence
  for x=-5,5,0.25 do
    for y=-5,5,0.25 do
      if y ~= 0 then
        assert(x%y == x-floor(x/y)*y)
      end
    end
  end
end

do --- jit constant RHS
  local y = 0
  for x=-100,123 do
    y = y + x%17
  end
  assert(y == 1777)
end

do --- jit constant LHS, with exit
  local y = 0
  for x=-100,123 do
    if x ~= 0 then
      y = y + 85%x
    end
  end
  assert(y == 2059)
end

do --- divide by zero
  local x = 1%0
  assert(x ~= x)
  x = floor(0/0)
  assert(x ~= x)
end
";
    let chunk = compile(source).unwrap();
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(vm.execute(&chunk).unwrap(), Vec::<Value>::new());
}

fn streamed_named(source: &[u8], config: CompilerConfig, fragment_size: usize) -> Chunk {
    let mut session = CompilerLoader::new(config).begin(LuaString::from("incremental-test"));
    for bytes in source.chunks(fragment_size) {
        session = match session.resume(LoadInput::Bytes(LuaString::from(bytes))) {
            LoadPoll::NeedInput(next) => next,
            _ => panic!("source terminated before end"),
        };
    }
    match session.resume(LoadInput::End) {
        LoadPoll::Loaded(chunk) => chunk,
        LoadPoll::Failed(error) => panic!("{}", String::from_utf8_lossy(error.as_bytes())),
        LoadPoll::NeedInput(_) => panic!("end needs input"),
    }
}

#[test]
fn incremental_compiler_emits_bytecode_before_end() {
    let mut parser = syntax::StreamingParser::default();
    let mut compilation = IncrementalCompilation::default();
    let name = LuaString::from("incremental-test");
    let config = CompilerConfig::default();
    for source in [b"local x=1; ".as_slice(), b"x=x+2; "] {
        assert!(matches!(
            parser.push(source),
            syntax::SourceParsePoll::NeedInput
        ));
        let before = compilation
            .compiler
            .as_ref()
            .map_or(0, |compiler| compiler.code.len());
        compilation.advance(
            &name,
            config,
            parser.completed_statements(),
            parser.line_starts(),
            false,
        );
        assert!(compilation.compiler.as_ref().unwrap().code.len() > before);
        assert!(!compilation.replay);
    }
    let chunk = compilation
        .finish(name.clone(), parser.end_owned().unwrap(), config)
        .unwrap();
    assert_eq!(
        chunk,
        compile_named(name, b"local x=1; x=x+2; ", config).unwrap()
    );
}

#[test]
fn incremental_compiler_matches_all_profiles_and_fragment_sizes() {
    let source =
        b"local x=1\r\nlocal function f(y) return x+y end\nfor i=1,3 do x=f(i) end\rreturn x";
    for profile in CompatibilityProfile::ALL {
        let config = CompilerConfig::new(profile);
        let expected = compile_named("incremental-test", source, config).unwrap();
        for size in 1..=source.len() {
            assert_eq!(
                streamed_named(source, config, size),
                expected,
                "{profile:?} fragment {size}"
            );
        }
    }
}

#[test]
fn incremental_compiler_replays_late_closure_writes() {
    let source = b"local x=1; sink(x, mutate()); local function later() x=2 end; return later";
    let mut parser = syntax::StreamingParser::default();
    let mut compilation = IncrementalCompilation::default();
    let name = LuaString::from("incremental-test");
    let config = CompilerConfig::default();
    assert!(matches!(
        parser.push(b"local x=1; sink(x, mutate()); "),
        syntax::SourceParsePoll::NeedInput
    ));
    compilation.advance(
        &name,
        config,
        parser.completed_statements(),
        parser.line_starts(),
        false,
    );
    assert!(compilation.emitted > 0);
    assert!(matches!(
        parser.push(b"local function later() x=2 end; return later "),
        syntax::SourceParsePoll::NeedInput
    ));
    compilation.advance(
        &name,
        config,
        parser.completed_statements(),
        parser.line_starts(),
        false,
    );
    assert!(compilation.replay);
    for size in 1..=source.len() {
        assert_eq!(
            streamed_named(source, config, size),
            compile_named(name.clone(), source, config).unwrap()
        );
    }
}

#[test]
fn incremental_compiler_preserves_trailing_labels_and_globals() {
    for source in [
        b"goto done; local x=1; ::done:: ::last::".as_slice(),
        b"local x=1; ::middle:: x=2; ::last::",
        b"local x=1; goto done; ::done:: return x",
        b"global<const> x=1; local function f() return x end; return f()",
    ] {
        for profile in CompatibilityProfile::ALL {
            let config = CompilerConfig::new(profile);
            let Ok(expected) = compile_named("incremental-test", source, config) else {
                continue;
            };
            for size in 1..=source.len() {
                assert_eq!(
                    streamed_named(source, config, size),
                    expected,
                    "{profile:?} fragment {size}"
                );
            }
        }
    }
}

#[test]
fn incremental_compiler_retries_compact_register_allocation() {
    let numbers = (0..110)
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!("local wide=select('#', {numbers}, select('#', {numbers})); return wide");
    let config = CompilerConfig::default();
    let expected = compile_named("incremental-test", &source, config).unwrap();
    let parsed = syntax::parse_bytes_with_config(
        source.as_bytes(),
        syntax::ParserConfig::new(config.compatibility_profile()),
    )
    .unwrap();
    assert!(matches!(
        compile_main(LuaString::from("incremental-test"), &parsed, config, false),
        Err(CompileError::RegisterExhausted { .. })
    ));
    for size in [1, 7, 31, source.len()] {
        assert_eq!(streamed_named(source.as_bytes(), config, size), expected);
    }
}

#[test]
fn incremental_compiler_defers_semantic_error_until_syntax_accepts_end() {
    let mut session = CompilerLoader::with_profile(CompatibilityProfile::Lua54)
        .begin(LuaString::from("incremental-test"));
    session = match session.resume(LoadInput::Bytes(LuaString::from(
        "local x <const> =1; x=2; ",
    ))) {
        LoadPoll::NeedInput(next) => next,
        _ => panic!("semantic failure must wait for syntax end"),
    };
    match session.resume(LoadInput::Bytes(LuaString::from("local ="))) {
        LoadPoll::Failed(message) => {
            assert!(!String::from_utf8_lossy(message.as_bytes()).contains("assign to const"));
        }
        LoadPoll::NeedInput(session) => match session.resume(LoadInput::End) {
            LoadPoll::Failed(message) => {
                assert!(!String::from_utf8_lossy(message.as_bytes()).contains("assign to const"));
            }
            _ => panic!("invalid syntax accepted"),
        },
        LoadPoll::Loaded(_) => panic!("invalid syntax accepted"),
    }
}

#[test]
fn incremental_compiler_defers_deep_trees_to_sized_stack() {
    let expression = (0..96).map(|_| "1").collect::<Vec<_>>().join("+");
    let source = format!("local x=1; value={expression}; return x");
    let config = CompilerConfig::default();
    let mut parser = syntax::StreamingParser::default();
    let mut compilation = IncrementalCompilation::default();
    let name = LuaString::from("incremental-test");
    assert!(matches!(
        parser.push(source.as_bytes()),
        syntax::SourceParsePoll::NeedInput
    ));
    compilation.advance(
        &name,
        config,
        parser.completed_statements(),
        parser.line_starts(),
        false,
    );
    assert!(compilation.replay);
    assert_eq!(
        streamed_named(source.as_bytes(), config, 7),
        compile_named(name, source.as_bytes(), config).unwrap()
    );
}

#[test]
fn incremental_compiler_preserves_semantic_diagnostics() {
    for source in [b"local x<const> =1; x=2".as_slice(), b"goto missing"] {
        let config = CompilerConfig::new(CompatibilityProfile::Lua54);
        let name = LuaString::from("incremental-test");
        let expected = compile_named(name.clone(), source, config)
            .unwrap_err()
            .diagnostic(&name, source);
        for size in 1..=source.len() {
            let mut session = CompilerLoader::new(config).begin(name.clone());
            for bytes in source.chunks(size) {
                session = match session.resume(LoadInput::Bytes(LuaString::from(bytes))) {
                    LoadPoll::NeedInput(next) => next,
                    LoadPoll::Failed(error) => panic!(
                        "semantic error before end: {}",
                        String::from_utf8_lossy(error.as_bytes())
                    ),
                    LoadPoll::Loaded(_) => panic!("invalid source loaded"),
                };
            }
            match session.resume(LoadInput::End) {
                LoadPoll::Failed(error) => assert_eq!(error, expected),
                _ => panic!("invalid source accepted"),
            }
        }
    }
}

#[test]
fn incremental_semantic_error_drops_later_deep_tree_on_sized_stack() {
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(|| {
            let expression = (0..2048).map(|_| "1").collect::<Vec<_>>().join("+");
            let source = format!("local x <const> =1; x=2; value={expression}");
            let name = LuaString::from("incremental-test");
            let config = CompilerConfig::new(CompatibilityProfile::Lua54);
            let mut session = CompilerLoader::new(config).begin(name.clone());
            for bytes in source.as_bytes().chunks(128) {
                session = match session.resume(LoadInput::Bytes(LuaString::from(bytes))) {
                    LoadPoll::NeedInput(next) => next,
                    _ => panic!("semantic error before end"),
                };
            }
            let expected = compile_named(name.clone(), source.as_bytes(), config)
                .unwrap_err()
                .diagnostic(&name, source.as_bytes());
            match session.resume(LoadInput::End) {
                LoadPoll::Failed(error) => assert_eq!(error, expected),
                _ => panic!("const assignment accepted"),
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// The compiler loads the global into a temporary and copies it into the call
/// register, so the shape scan must follow the `Move` back to the load.
#[test]
fn constructor_shape_survives_global_setmetatable_copy() {
    let chunk =
        compile("local mt={} for i=1,6 do local o=setmetatable({a=1},mt) o.b=i end").unwrap();
    let pc = chunk
        .code
        .iter()
        .rposition(|instruction| instruction.opcode() == Ok(OpCode::NewTable))
        .unwrap();
    let shape = chunk.constructor_shape(pc).expect("constructor shape");
    let keys: Vec<_> = shape
        .keys()
        .iter()
        .map(|key| key.as_bytes().to_vec())
        .collect();
    assert_eq!(keys, vec![b"a".to_vec(), b"b".to_vec()]);
    // A call through some other global is still not a shape.
    let other = compile("local mt={} for i=1,6 do local o=other({a=1},mt) o.b=i end").unwrap();
    let pc = other
        .code
        .iter()
        .rposition(|instruction| instruction.opcode() == Ok(OpCode::NewTable))
        .unwrap();
    assert!(other.constructor_shape(pc).is_none());
}

/// The store after `setmetatable` reaches the filled write site once the
/// constructor shape is known, for the global and for a local alias.
#[test]
fn store_site_fills_after_setmetatable_constructor() {
    for alias in ["", "local setmetatable = setmetatable "] {
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
            let source = format!(
                "{alias}local mt = {{}} local function run() for i = 1, 6 do local o = setmetatable({{a = 1}}, mt) o.b = i end end run()"
            );
            let config = CompilerConfig::new(profile);
            let chunk = compile_with_config(&source, config).unwrap();
            let mut vm = Vm::with_profile(profile);
            install_base_with_profile(vm.runtime_mut(), profile).unwrap();
            vm.execute(&chunk).unwrap();
            // `execute` runs a clone of the chunk, whose prototypes are
            // shared, so the loop lives in one.
            let chunk = &chunk.prototypes[0];
            let pc = chunk
                .code
                .iter()
                .rposition(|instruction| instruction.opcode() == Ok(OpCode::SetField))
                .unwrap();
            let site = chunk.store_site(pc).expect("write sites installed");
            assert!(
                site.load_pair().is_some_and(|(guard, _)| guard != 0),
                "{profile:?} {alias:?}: write site not filled"
            );
        }
    }
}

/// `NewTable` carries the constructor's item counts, which presize the array
/// part: positional items in `b`, keyed items in `c`, a trailing multiple
/// result not counted, each capped at what a byte holds.
#[test]
fn new_table_carries_the_constructors_item_counts() {
    fn collect(chunk: &crate::bytecode::Chunk, out: &mut Vec<(u8, u8)>) {
        out.extend(
            chunk
                .code
                .iter()
                .filter(|instruction| instruction.opcode() == Ok(OpCode::NewTable))
                .map(|instruction| (instruction.b(), instruction.c())),
        );
        for prototype in &chunk.prototypes {
            collect(prototype, out);
        }
    }
    let hints = |source: &str| {
        let chunk = compile(source).unwrap();
        let mut found = Vec::new();
        collect(&chunk, &mut found);
        found
    };
    assert_eq!(hints("return {}"), [(0, 0)]);
    assert_eq!(hints("return {1, 2, 3}"), [(3, 0)]);
    assert_eq!(hints("return {a = 1, b = 2}"), [(0, 2)]);
    assert_eq!(hints("return {1, x = 2, [3] = 4, 5}"), [(2, 2)]);
    assert_eq!(
        hints("local function f() return 1, 2 end return {f()}"),
        [(0, 0)]
    );
    assert_eq!(
        hints("local function f() return 1, 2 end return {1, f()}"),
        [(1, 0)]
    );
    assert_eq!(hints("return function(...) return {...} end"), [(0, 0)]);
    assert_eq!(hints("return function(...) return {1, ...} end"), [(1, 0)]);
    // Each constructor counts its own items, nested ones included.
    assert_eq!(hints("return {{1}, {2, 3}}"), [(2, 0), (1, 0), (2, 0)]);
    let many = format!("return {{{}}}", vec!["1"; 300].join(","));
    assert_eq!(hints(&many), [(255, 0)]);
}
