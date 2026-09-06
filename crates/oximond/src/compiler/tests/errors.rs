use super::*;

#[test]
fn lua54_lua_close_handlers_unwind_with_replacement_errors() {
    let lua54 = CompilerConfig::new(CompatibilityProfile::Lua54);
    run_with_config(
        "local first, second; \
             local function failing() \
               local a<close> = setmetatable({}, {__close = function(_, err) second = err end}); \
               local b<close> = setmetatable({}, {__close = function(_, err) first = err; error('replacement') end}); \
               error('original') \
             end; \
             local ok, err = pcall(failing); \
             assert(not ok and first ~= nil and second ~= first and err == second)",
        lua54,
    );

    let chunk = compile_with_config(
            "local function failing() \
               local value<close> = setmetatable({}, {__close = function() error('replacement') end}); \
               error('original') \
             end; failing()",
            lua54,
        )
        .unwrap();
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
    install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua54).unwrap();
    let error = vm.execute(&chunk).unwrap_err();
    assert!(error.to_string().contains("replacement"));
}

#[test]
fn nested_coroutines_report_normal_and_preserve_error_objects() {
    run("local outer; \
             outer=coroutine.create(function() \
               local child=coroutine.create(function() \
                 assert(coroutine.status(outer)=='normal'); coroutine.yield('x') \
               end); \
               local ok,value=coroutine.resume(child); \
               assert(ok and value=='x'); return coroutine.status(child) \
             end); \
             local ok,status=coroutine.resume(outer); \
             assert(ok and status=='suspended'); \
             local marker={}; \
             local wrapped=coroutine.wrap(function() error(marker) end); \
             local ok,error_object=pcall(wrapped); \
             assert(not ok and error_object==marker)");
}

#[test]
fn compiler_loader_reports_a_terminal_failure_on_end() {
    let loader = CompilerLoader::default();
    let session = loader.begin(LuaString::from("broken"));
    let session = match session.resume(LoadInput::Bytes(LuaString::from("local ="))) {
        LoadPoll::NeedInput(session) => session,
        LoadPoll::Loaded(_) | LoadPoll::Failed(_) => panic!("fragment must not compile early"),
    };
    assert!(matches!(
        session.resume(LoadInput::End),
        LoadPoll::Failed(_)
    ));
}

#[test]
fn compiler_loader_fails_when_the_second_one_byte_reader_pull_releases_an_invalid_start() {
    let loader = CompilerLoader::default();
    let session = loader.begin(LuaString::from("reader"));
    let session = match session.resume(LoadInput::Bytes(LuaString::from("*"))) {
        LoadPoll::NeedInput(session) => session,
        LoadPoll::Loaded(_) | LoadPoll::Failed(_) => {
            panic!("a lone '*' needs one byte of lexical lookahead")
        }
    };
    match session.resume(LoadInput::Bytes(LuaString::from("a"))) {
        LoadPoll::Failed(error) => {
            assert_eq!(
                error.as_bytes(),
                b"[string \"reader\"]:1: expected expression near '*'"
            );
        }
        LoadPoll::NeedInput(_) | LoadPoll::Loaded(_) => {
            panic!("second one-byte reader pull must be terminal")
        }
    }
}

#[test]
fn lua51_dynamic_syntax_diagnostics_name_the_near_token_and_eof_line() {
    let config = CompilerConfig::new(CompatibilityProfile::Lua51);
    for (source, expected) in [
        (
            b"break label" as &[u8],
            b"[string \"break label\"]:1: break outside a loop near 'label'" as &[u8],
        ),
        (
            b"  local a = {4\n\n" as &[u8],
            b"[string \"  local a = {4...\"]:3: unexpected token in expression near '<eof>'"
                as &[u8],
        ),
    ] {
        let error = compile_named(LuaString::from(source), source, config).unwrap_err();
        let CompileError::Syntax(error) = error else {
            panic!("invalid source must fail in the parser");
        };
        assert_eq!(error.diagnostic().as_bytes(), expected);
    }
}
