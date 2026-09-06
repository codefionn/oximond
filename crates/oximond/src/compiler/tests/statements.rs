use super::*;

#[test]
fn implicit_numeric_for_step_keeps_modern_integer_loops_at_the_bounds() {
    let config = CompilerConfig::new(CompatibilityProfile::Lua53);
    for source in [
        "local edge=math.maxinteger; local count=0; for i=edge-1,edge do count=count+1; if count>5 then break end end; return count",
        "local edge=math.mininteger; local count=0; for i=edge,edge+1 do count=count+1; if count>5 then break end end; return count",
    ] {
        let chunk = compile_with_config(source, config).unwrap();
        let mut vm = Vm::with_profile(CompatibilityProfile::Lua53);
        install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua53).unwrap();
        assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Integer(2)]);
    }
}

#[test]
fn lua52_globals_follow_the_lexical_environment() {
    let lua52 = CompilerConfig::new(CompatibilityProfile::Lua52);
    let chunk = compile_with_config("return _ENV", lua52).unwrap();
    assert_eq!(chunk.upvalues.len(), 1);
    assert_eq!(chunk.upvalues[0].name.as_bytes(), b"_ENV");
    assert!(matches!(chunk.upvalues[0].source, UpvalueSource::Detached));
    let vm = run_with_config(
        "local root = _ENV; \
             local private = { value = 40 }; root.private = private; \
             do \
               local _ENV = private; \
               answer = value + 2; \
               local function read() return value end; \
               local function factory() return function() return value end end; \
               local descendant = factory(); \
               root.before = read(); \
               _ENV = { value = 7 }; \
               root.after = read(); root.descendant = descendant(); \
               changed = value; root.replacement = _ENV \
             end",
        lua52,
    );
    assert_eq!(vm.runtime().get_global("before"), Value::Number(40.0));
    assert_eq!(vm.runtime().get_global("after"), Value::Number(7.0));
    assert_eq!(vm.runtime().get_global("descendant"), Value::Number(7.0));
    let Value::Table(private) = vm.runtime().get_global("private") else {
        panic!("private environment was not published")
    };
    assert_eq!(
        vm.runtime()
            .table_get(private, &Value::from("answer"))
            .unwrap(),
        Value::Number(42.0)
    );
    assert_eq!(
        vm.runtime()
            .table_get(private, &Value::from("changed"))
            .unwrap(),
        Value::Nil
    );
    let Value::Table(replacement) = vm.runtime().get_global("replacement") else {
        panic!("replacement environment was not published")
    };
    assert_eq!(
        vm.runtime()
            .table_get(replacement, &Value::from("changed"))
            .unwrap(),
        Value::Number(7.0)
    );

    let lua51 = CompilerConfig::new(CompatibilityProfile::Lua51);
    let vm = run_with_config("local _ENV = {}; old_profile_global = 9", lua51);
    assert_eq!(
        vm.runtime().get_global("old_profile_global"),
        Value::Number(9.0)
    );
}

#[test]
fn lua52_string_format_uses_tostring_metamethods() {
    let lua52 = CompilerConfig::new(CompatibilityProfile::Lua52);
    let vm = run_with_config(
        "local converted = setmetatable({}, { \
                 __tostring = function() return 'converted' end \
             }); \
             formatted = string.format('%s:%s:%s', nil, true, converted); \
             local bad = setmetatable({}, { __tostring = function() return {} end }); \
             format_ok, format_error = pcall(string.format, '%s', bad)",
        lua52,
    );
    assert_eq!(
        vm.runtime().get_global("formatted"),
        Value::from("nil:true:converted")
    );
    assert_eq!(vm.runtime().get_global("format_ok"), Value::Boolean(false));
    let Value::String(error) = vm.runtime().get_global("format_error") else {
        panic!("string.format returned a non-string error")
    };
    assert!(
        error
            .as_bytes()
            .windows(b"'__tostring' must return a string".len())
            .any(|window| window == b"'__tostring' must return a string")
    );
}

#[test]
fn lua54_close_locals_are_marked_after_initializers_and_closed_on_scope_exit() {
    let lua54 = CompilerConfig::new(CompatibilityProfile::Lua54);
    let chunk = compile_with_config("do local resource<close> = open() end", lua54).unwrap();
    let mark = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::MarkClose))
        .expect("close declaration emits its marker");
    let close = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::CloseLocal))
        .expect("scope exit closes the local");
    assert!(mark < close);

    let chunk = compile_with_config("local resource<close> = open(); return 1", lua54).unwrap();
    let return_pc = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::Return))
        .expect("return instruction exists");
    assert_eq!(chunk.code[return_pc - 1].opcode(), Ok(OpCode::CloseLocal));

    let chunk =
        compile_with_config("local resource<close> = open(); return finish()", lua54).unwrap();
    let call = chunk
        .debug
        .call_sites()
        .last()
        .expect("tail call has debug metadata");
    assert_eq!(chunk.code[call.pc()].opcode(), Ok(OpCode::TailCall));
}

#[test]
fn lua52_gotos_jump_forward_and_backward_and_validate_labels() {
    let lua52 = CompilerConfig::new(CompatibilityProfile::Lua52);
    let vm = run_with_config(
        "local n = 0; goto start; ::skip:: n = 99; ::start:: n = n + 1; if n < 3 then goto start end; result = n",
        lua52,
    );
    assert_eq!(vm.runtime().get_global("result"), Value::Number(3.0));
    assert!(matches!(
        compile_with_config("goto missing", lua52),
        Err(CompileError::UndefinedLabel { .. })
    ));
    assert!(matches!(
        compile_with_config("::same:: ::same::", lua52),
        Err(CompileError::DuplicateLabel { .. })
    ));
    assert!(compile_with_config("::same:: do ::same:: end", lua52).is_ok());
    assert!(
        compile_with_config(
            "::same:: do ::same:: end",
            CompilerConfig::new(CompatibilityProfile::Lua53)
        )
        .is_ok()
    );
    assert!(matches!(
        compile_with_config(
            "::same:: do ::same:: end",
            CompilerConfig::new(CompatibilityProfile::Lua54)
        ),
        Err(CompileError::DuplicateLabel { .. })
    ));
    assert!(matches!(
        compile_with_config("goto later; local x = 1; ::later:: print(x)", lua52),
        Err(CompileError::GotoIntoLocalScope { .. })
    ));
    let vm = run_with_config(
        "local kept; do local value = 7; kept = function() return value end; \
             goto outside; value = 9 end; ::outside:: result = kept()",
        lua52,
    );
    assert_eq!(vm.runtime().get_global("result"), Value::Number(7.0));
}

#[test]
fn lua55_const_globals_and_named_varargs_are_readonly_and_preserve_varargs() {
    let lua55 = CompilerConfig::new(CompatibilityProfile::Lua55);
    let vm = run_with_config(
        "global<const> answer = 42; global *; later = answer; \
             function collect(head, ... extras) return extras.n, extras[1], extras[2], head, ... end; \
             global function twice(value) return value * 2 end; doubled = twice(21); \
             n, first, second, head, extra_first, extra_second = collect(9, 3, 4)",
        lua55,
    );
    assert_eq!(vm.runtime().get_global("answer"), Value::Integer(42));
    assert_eq!(vm.runtime().get_global("later"), Value::Integer(42));
    assert_eq!(vm.runtime().get_global("doubled"), Value::Integer(42));
    assert_eq!(vm.runtime().get_global("n"), Value::Integer(2));
    assert_eq!(vm.runtime().get_global("first"), Value::Integer(3));
    assert_eq!(vm.runtime().get_global("second"), Value::Integer(4));
    assert_eq!(vm.runtime().get_global("head"), Value::Integer(9));
    assert_eq!(vm.runtime().get_global("extra_first"), Value::Integer(3));
    assert_eq!(vm.runtime().get_global("extra_second"), Value::Integer(4));
    assert!(matches!(
        compile_with_config("global a<const>; a = 1; a = 2", lua55),
        Err(CompileError::AssignToReadOnly { .. })
    ));
    assert!(matches!(
        compile_with_config("function f(... extras) extras = {} end", lua55),
        Err(CompileError::AssignToReadOnly { .. })
    ));
    let vm = run_with_config(
        "global<const> assert; global result; \
             local function set_result() do global result; result = 42 end; \
             assert(_ENV.result == 42) end; set_result()",
        lua55,
    );
    assert_eq!(vm.runtime().get_global("result"), Value::Integer(42));
    let vm = run_with_config(
        "global X, local_result, global_result, recursive_result, env_result; X = 20; \
             do local X = 10; do global X; X = 30 end; local_result = X end; \
             global_result = _ENV.X; \
             do local foo = 20; do global function foo(x) \
               if x == 0 then return 1 else return 2 * foo(x - 1) end \
             end; recursive_result = foo(3) end end; \
             do global *; local function env() return _ENV end; env_result = env() == _ENV end; \
             do local initialized = 42; do global initialized = initialized end end",
        lua55,
    );
    assert_eq!(vm.runtime().get_global("local_result"), Value::Integer(10));
    assert_eq!(vm.runtime().get_global("global_result"), Value::Integer(30));
    assert_eq!(
        vm.runtime().get_global("recursive_result"),
        Value::Integer(8)
    );
    assert_eq!(vm.runtime().get_global("env_result"), Value::Boolean(true));
    assert_eq!(vm.runtime().get_global("initialized"), Value::Integer(42));
    let vm = run_with_config(
        "global result; global<const> *; local x = 1; x = 2; \
             local function set() x = 3 end; set(); result = x",
        lua55,
    );
    assert_eq!(vm.runtime().get_global("result"), Value::Integer(3));
    assert!(matches!(
        compile_with_config("global _ENV, a; a = 10", lua55),
        Err(CompileError::UndeclaredGlobal { name, .. }) if name == "a"
    ));
    assert!(matches!(
        compile_with_config("global<const> foo; function foo() end", lua55),
        Err(CompileError::AssignToReadOnly { name, .. }) if name == "foo"
    ));
}

#[test]
fn lua55_global_all_does_not_hide_a_nested_upvalue() {
    let chunk = compile_with_config(
        "global *; local function outer(value) return function() return value end end",
        CompilerConfig::new(CompatibilityProfile::Lua55),
    )
    .unwrap();
    let outer = chunk.prototypes.first().expect("outer function prototype");
    let inner = outer.prototypes.first().expect("inner function prototype");
    assert!(
        inner
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::GetUpvalue))
    );
}

#[test]
fn lua55_initialized_globals_reject_existing_values_in_the_active_environment() {
    let lua55 = CompilerConfig::new(CompatibilityProfile::Lua55);
    let chunk = compile_with_config("global print = 10", lua55).unwrap();
    assert!(chunk.code.iter().any(|instruction| matches!(
        instruction.opcode(),
        Ok(OpCode::DeclareGlobal | OpCode::DeclareTable)
    )));
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua55);
    install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua55).unwrap();
    let error = vm.execute(&chunk).unwrap_err();
    assert!(error.to_string().contains("global 'print' already defined"));

    let chunk = compile_with_config(
        "global pcall; \
             local env = {AA = false}; \
             local function initialize(_ENV) global AA = 10 end; \
             return pcall(initialize, env)",
        lua55,
    )
    .unwrap();
    assert!(
        chunk.prototypes[0]
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::DeclareTable))
    );
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua55);
    install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua55).unwrap();
    let results = vm.execute(&chunk).unwrap();
    assert_eq!(results.first(), Some(&Value::Boolean(false)));
    let Some(Value::String(message)) = results.get(1) else {
        panic!("pcall returns the declaration error message");
    };
    assert!(String::from_utf8_lossy(message.as_bytes()).contains("global 'AA' already defined"));
}

#[test]
fn reclaimed_temporaries_do_not_keep_weak_table_keys_alive() {
    run("a=setmetatable({}, {__mode='k'}); \
             local dead={}; a[dead]=1; dead=nil; \
             collectgarbage(); assert(next(a)==nil)");
}

// The loop ends at the first collection, which a 1 MiB nursery reaches
// after some tens of thousands of these strings: the bound only stops a
// loop that would never end.
#[test]
fn temporary_while_conditions_do_not_keep_weak_values_alive() {
    run(
        "local n=0; local weak=setmetatable({[1]={}}, {__mode='kv'}); \
             while weak[1] do \
               local garbage=n..n..n..n; n=n+1; \
               assert(n < 200000) \
             end; \
             assert(n > 0)",
    );
}

#[test]
fn captured_locals_are_closed_and_cleared_when_a_do_scope_ends() {
    run(
        "collectgarbage('stop'); finalized=0; local proxy=newproxy(true); \
             do local captured=proxy; \
               getmetatable(captured).__gc=function(object) \
                 assert(getmetatable(object)==getmetatable(captured)); \
                 finalized=finalized+1 \
               end \
             end; \
             proxy=nil; collectgarbage(); assert(finalized==1)",
    );
}

#[test]
fn straight_line_locals_reuse_dead_temporary_registers() {
    let mut source = String::new();
    for index in 0..96 {
        writeln!(source, "local value{index} = ({{1, 2, 3}})[2] + 4").unwrap();
    }
    source.push_str("assert(value0 == 6 and value95 == 6)");

    let chunk = compile(&source).unwrap();
    assert!(chunk.register_count < 128);
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&chunk).unwrap();
}

#[test]
fn single_local_arithmetic_assignment_writes_directly_to_its_register() {
    let chunk = compile("local sum=0; for i=1,10 do sum=sum+i end; return sum").unwrap();
    let prep = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForPrep))
        .unwrap();
    let loop_pc = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForLoop))
        .unwrap();
    let body = chunk.code[prep + 1..loop_pc]
        .iter()
        .map(|instruction| instruction.opcode().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(body, vec![OpCode::Add]);
    let mut vm = Vm::new();
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(55.0)]);
}

fn moves_before_first_call(chunk: &crate::bytecode::Chunk) -> usize {
    chunk
        .code
        .iter()
        .take_while(|instruction| instruction.opcode() != Ok(OpCode::Call))
        .filter(|instruction| instruction.opcode() == Ok(OpCode::Move))
        .count()
}

#[test]
fn call_operands_are_evaluated_into_their_call_slots() {
    // The callee and the constant arguments are loaded straight into the
    // call window, so only the local `source` is copied.
    let chunk =
        compile("local source = 'abc'; local found = string.find(source, 'b', 1, true)").unwrap();
    assert_eq!(moves_before_first_call(&chunk), 1);
    let call_pc = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::Call))
        .expect("four arguments use Call");
    let call = chunk.code[call_pc];
    let loads = chunk.code[..call_pc]
        .iter()
        .filter(|instruction| {
            instruction.opcode() == Ok(OpCode::LoadConstant)
                && instruction.a() > call.a()
                && instruction.a() <= call.a() + call.b()
        })
        .count();
    assert_eq!(loads, 3);
    assert_eq!(chunk.validate(), Ok(()));
}

#[test]
fn operands_survive_a_later_call_in_the_argument_list() {
    // `x` and `10` sit in their call slots while `g()` runs a frame whose
    // locals would reuse any registers shared with this window.
    let source = "local function g() local a, b, c, d, e, f = 1, 2, 3, 4, 5, 6; return 7 end \
                  local function f(...) return select('#', ...), ... end \
                  local n, a, b, c, d = f('x', 10, g(), 'y'); return n, a, b, c, d";
    let chunk = compile(source).unwrap();
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![
            Value::Number(4.0),
            Value::from("x"),
            Value::Number(10.0),
            Value::Number(7.0),
            Value::from("y"),
        ]
    );
}

#[test]
fn operands_that_other_code_reads_are_copied_into_the_window() {
    // `t.x` feeds the `t.x + 1` operand as well, so its register is not a
    // scratch value of the call slot.
    let source = "local t = {x = 5}; local function f(a, b) return a * 10 + b end \
                  return f(t.x, t.x + 1), f(t.x + 1, t.x)";
    let chunk = compile(source).unwrap();
    let mut vm = Vm::new();
    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::Number(56.0), Value::Number(65.0)]
    );
}

#[test]
fn single_local_call_assignment_writes_directly_to_its_register() {
    let chunk = compile(
        "local identity=function(value) return value end; local result=0; result=identity(7)",
    )
    .unwrap();
    let call_pc = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::CallOne))
        .expect("one-argument call uses CallOne");
    assert_ne!(chunk.code[call_pc + 1].opcode(), Ok(OpCode::Move));
    assert_eq!(chunk.validate(), Ok(()));
}

#[test]
fn direct_local_call_assignment_allows_callee_and_argument_aliases() {
    let callee_chunk =
        compile("local value=function(number) return number+1 end; value=value(41); return value")
            .unwrap();
    let callee_call = callee_chunk
        .code
        .iter()
        .copied()
        .find(|instruction| instruction.opcode() == Ok(OpCode::CallOne))
        .expect("callee assignment has CallOne");
    assert_eq!(callee_call.a(), callee_call.b());
    let mut vm = Vm::new();
    assert_eq!(
        vm.execute(&callee_chunk).unwrap(),
        vec![Value::Number(42.0)]
    );

    let argument_chunk = compile(
            "local identity=function(number) return number end; local value=42; value=identity(value); return value",
        )
        .unwrap();
    let argument_call = argument_chunk
        .code
        .iter()
        .copied()
        .find(|instruction| instruction.opcode() == Ok(OpCode::CallOne))
        .expect("argument assignment has CallOne");
    assert_eq!(argument_call.a(), argument_call.c());
    let mut vm = Vm::new();
    assert_eq!(
        vm.execute(&argument_chunk).unwrap(),
        vec![Value::Number(42.0)]
    );
}

#[test]
fn captured_local_call_assignment_writes_directly_to_its_register() {
    let chunk = compile(
        "local function outer() \
               local value=0; \
               local identity=function(number) return number end; \
               local function read() return value end; \
               value=identity(42); \
               return read() \
             end; \
             return outer()",
    )
    .unwrap();
    let outer = &chunk.prototypes[0];
    let call_pc = outer
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::CallOne))
        .expect("captured local assignment has CallOne");

    let call = outer.code[call_pc];
    let next = outer.code[call_pc + 1];
    assert!(
        next.opcode() != Ok(OpCode::Move) || next.b() != call.a(),
        "the call result must not be copied into the captured local"
    );
    let mut vm = Vm::new();
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(42.0)]);
}

#[test]
fn only_single_plain_local_assignments_retarget_call_results() {
    let chunk = compile(
        "local identity=function(value) return value end; \
             local first,second=0,0; first,second=identity(1),2; \
             local values={}; values[1]=identity(3); \
             global=identity(4)",
    )
    .unwrap();
    let calls = chunk
        .code
        .iter()
        .enumerate()
        .filter(|(_, instruction)| instruction.opcode() == Ok(OpCode::CallOne))
        .collect::<Vec<_>>();

    assert_eq!(calls.len(), 3);
    let (multiple_call_pc, multiple_call) = calls[0];
    assert!(
        chunk.code[multiple_call_pc + 1..]
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::Move)
                && instruction.b() == multiple_call.a())
    );
    let (indexed_call_pc, indexed_call) = calls[1];
    assert_eq!(
        chunk.code[indexed_call_pc + 1].opcode(),
        Ok(OpCode::SetTable)
    );
    assert_eq!(chunk.code[indexed_call_pc + 1].c(), indexed_call.a());
    let (global_call_pc, global_call) = calls[2];
    assert_eq!(
        chunk.code[global_call_pc + 1].opcode(),
        Ok(OpCode::SetGlobal)
    );
    assert_eq!(chunk.code[global_call_pc + 1].a(), global_call.a());

    let upvalue_chunk = compile(
        "local value=0; \
             local identity=function(number) return number end; \
             local function set() value=identity(5) end; \
             set()",
    )
    .unwrap();
    let set = upvalue_chunk
        .prototypes
        .iter()
        .find(|prototype| {
            prototype
                .code
                .iter()
                .any(|instruction| instruction.opcode() == Ok(OpCode::CallOne))
        })
        .expect("nested upvalue assignment has CallOne");
    let upvalue_call_pc = set
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::CallOne))
        .expect("nested upvalue assignment has CallOne");
    let upvalue_call = set.code[upvalue_call_pc];
    assert_eq!(
        set.code[upvalue_call_pc + 1].opcode(),
        Ok(OpCode::SetUpvalue)
    );
    assert_eq!(set.code[upvalue_call_pc + 1].a(), upvalue_call.a());
}

#[test]
fn numeric_for_clears_body_locals_before_its_latch() {
    let chunk = compile("for i=1,2 do local table_value={value=i}; table_value.n=i end").unwrap();
    let local = chunk
        .debug
        .local_variables()
        .iter()
        .find(|local| local.name().as_bytes() == b"table_value")
        // A local whose initializer ends the body has an empty range and no
        // entry, so the body uses it once more.
        .expect("body-local register is recorded for debugging");
    let latch = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForLoop))
        .expect("numeric loop has a latch");
    assert!(local.start_pc() < latch);
    // The expired local is not cleared. It is dead at the latch, so the live
    // register count there stops below it and the collector does not root it.
    assert!(
        chunk.code[local.end_pc()..latch]
            .iter()
            .all(|instruction| !matches!(
                instruction.opcode(),
                Ok(OpCode::LoadNil | OpCode::ClearRange)
            ))
    );
    assert!(chunk.live_register_count(latch) <= usize::from(local.register()));
    assert!(!chunk.live_registers(latch).contains(local.register()));
}

#[test]
fn table_constructor_temporary_registers_do_not_survive_into_a_loop() {
    let source = "local weak = {[1] = {}}; \
             setmetatable(weak, {__mode = 'kv'}); \
             local allocations = 0; \
             while weak[1] and allocations < 3 do \
               local garbage = allocations..allocations..allocations..allocations; \
               collectgarbage(); allocations = allocations + 1 \
             end; \
             assert(weak[1] == nil)";
    let chunk = compile(source).unwrap();
    let get_weak_value = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::GetTable))
        .expect("while condition reads the weak value");
    let (condition_jump, jump) = chunk
        .code
        .iter()
        .enumerate()
        .skip(get_weak_value)
        .rfind(|(_, instruction)| instruction.opcode() == Ok(OpCode::JumpIfFalse))
        .expect("while condition branches on the weak value");
    // Nothing clears the condition's temporaries. They are dead in the body,
    // so the collector does not root them.
    assert!(!matches!(
        chunk.code[condition_jump + 1].opcode(),
        Ok(OpCode::LoadNil | OpCode::ClearRange)
    ));
    let live = chunk.live_registers(condition_jump + 1);
    assert!(!live.contains(chunk.code[get_weak_value].a()));
    assert!(!live.contains(jump.a()));

    run(source);
}

#[test]
fn branch_local_register_assignments_do_not_escape_their_path() {
    run("local first = 11; local second = 22; \
             if false then local hidden = 33; first = hidden end; \
             assert(first == 11 and second == 22)");
}

#[test]
fn compiler_config_applies_profile_specific_empty_statement_rules() {
    let lua51 = CompilerConfig::new(CompatibilityProfile::Lua51);
    let luajit = CompilerConfig::new(CompatibilityProfile::LuaJit21);

    assert!(compile_with_config("a = 1; b = 2", lua51).is_ok());
    assert!(compile_with_config(";", lua51).is_err());
    assert!(compile_with_config("a = 1;;", lua51).is_err());
    assert!(compile_with_config(";", luajit).is_ok());
    assert!(
        compile(";").is_ok(),
        "default compilation must remain LuaJIT"
    );
}

#[test]
fn named_compile_formats_lua_syntax_diagnostics_without_losing_token_bytes() {
    let error = compile_named(
        LuaString::from("@broken.lua"),
        b"local value = 1\r\n\xffbad",
        CompilerConfig::default(),
    )
    .unwrap_err();
    let CompileError::Syntax(error) = error else {
        panic!("invalid source must fail in the parser");
    };
    assert_eq!(
        error.diagnostic().as_bytes(),
        b"broken.lua:2: unexpected character near '\xff'"
    );
}

#[test]
fn emits_local_debug_ranges_for_parameters_arg_scopes_and_shadowing() {
    let root = compile(
        "local function inspect(first, ...) \
             local outer = first; \
             do local hidden = 7; outer = hidden end; \
             local outer = 9; \
             return outer \
             end",
    )
    .unwrap();
    let function = &root.prototypes[0];
    let locals = function.debug.local_variables();

    assert_eq!(locals[0].name().as_bytes(), b"first");
    assert_eq!(locals[0].start_pc(), 0);
    assert_eq!(locals[1].name().as_bytes(), b"arg");
    assert_eq!(locals[1].start_pc(), 0);

    let hidden = locals
        .iter()
        .find(|local| local.name().as_bytes() == b"hidden")
        .expect("block local metadata exists");
    let visible_at_hidden = function
        .debug
        .locals_at(hidden.start_pc())
        .map(|local| local.name().as_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        visible_at_hidden,
        vec![
            b"first" as &[u8],
            b"arg" as &[u8],
            b"outer" as &[u8],
            b"hidden" as &[u8],
        ]
    );

    let second_outer = locals
        .iter()
        .filter(|local| local.name().as_bytes() == b"outer")
        .max_by_key(|local| local.start_pc())
        .expect("shadowing binding metadata exists");
    assert_eq!(
        function
            .debug
            .locals_at(second_outer.start_pc())
            .filter(|local| local.name().as_bytes() == b"outer")
            .count(),
        1
    );
    assert!(second_outer.end_pc() <= function.code.len());
}

#[test]
fn return_calls_emit_proper_tail_call_variants() {
    let chunk = compile(
        b"local function fixed(f, x) return f(x) end \
              local function var(f, ...) return f(...) end \
              local function open(f, g) return f(g()) end",
    )
    .unwrap();
    let opcodes: Vec<Vec<OpCode>> = chunk
        .prototypes
        .iter()
        .map(|prototype| {
            prototype
                .code
                .iter()
                .map(|instruction| instruction.opcode().unwrap())
                .collect()
        })
        .collect();
    assert!(opcodes[0].contains(&OpCode::TailCall));
    assert!(opcodes[1].contains(&OpCode::TailCallVararg));
    assert!(opcodes[2].contains(&OpCode::TailCallOpen));
    for prototype in &chunk.prototypes {
        let tail_pc = prototype
            .code
            .iter()
            .position(|instruction| {
                matches!(
                    instruction.opcode(),
                    Ok(OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen)
                )
            })
            .expect("prototype has a tail call");
        assert_eq!(prototype.code[tail_pc + 1].opcode(), Ok(OpCode::ReturnOpen));
    }
}

#[test]
fn single_value_return_uses_the_expression_register_directly() {
    let chunk = compile("local function increment(value) return value + 1 end").unwrap();
    let prototype = &chunk.prototypes[0];
    let opcodes = prototype
        .code
        .iter()
        .map(|instruction| instruction.opcode().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(opcodes, vec![OpCode::AddK, OpCode::Return, OpCode::Halt]);
}

#[test]
fn compilation_records_named_operands_for_runtime_diagnostics() {
    let chunk = compile_named(
            "diagnostics.lua",
            b"local local_value = 'x'; local sink = global_value.field; return local_value + global_value", CompilerConfig::default())
        .unwrap();
    let sites: Vec<_> = chunk
        .debug
        .diagnostic_sites()
        .iter()
        .map(|site| {
            (
                site.name().as_bytes().to_vec(),
                site.kind(),
                chunk.code[site.pc()].opcode().unwrap(),
            )
        })
        .collect();
    assert!(sites.contains(&(b"global_value".to_vec(), CallKind::Global, OpCode::GetField)));
    assert!(sites.contains(&(b"local_value".to_vec(), CallKind::Local, OpCode::Add)));
    assert!(sites.contains(&(b"global_value".to_vec(), CallKind::Global, OpCode::Add)));
}

#[test]
fn compilation_deduplicates_repeated_operand_diagnostic_metadata() {
    let chunk = compile_named(
        "diagnostics.lua",
        b"local value = 1; return value + value",
        CompilerConfig::default(),
    )
    .unwrap();
    let add_pc = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::Add))
        .expect("compiled expression has an add instruction");
    let sites: Vec<_> = chunk
        .debug
        .diagnostic_sites()
        .iter()
        .filter(|site| site.pc() == add_pc)
        .collect();
    assert_eq!(sites.len(), 1);
    assert_eq!(sites[0].name().as_bytes(), b"value");
    assert_eq!(sites[0].kind(), CallKind::Local);
}

#[test]
fn captures_only_used_outer_locals_and_keeps_descriptor_names_in_use_order() {
    let chunk = compile(
        "local a, b; local function use_a() return a end; \
             local function unused() return 1 end; \
             local function make() local a, b; return function() return a, b end end",
    )
    .unwrap();
    let use_a = &chunk.prototypes[0];
    let unused = &chunk.prototypes[1];
    let make = &chunk.prototypes[2];
    assert_eq!(
        use_a
            .upvalues
            .iter()
            .map(|upvalue| upvalue.name.as_bytes())
            .collect::<Vec<_>>(),
        vec![b"a"]
    );
    assert_eq!(
        unused.upvalues,
        [] as [crate::bytecode::UpvalueDescriptor; 0]
    );
    let inner = &make.prototypes[0];
    assert_eq!(
        inner
            .upvalues
            .iter()
            .map(|upvalue| upvalue.name.as_bytes())
            .collect::<Vec<_>>(),
        vec![b"a".as_slice(), b"b".as_slice()]
    );
}

#[test]
fn forwards_descendant_only_captures_once_and_respects_shadowing() {
    let chunk = compile(
        "local outer; \
             local function forward() return function() return outer end end; \
             local function shadow() local outer = 2; return function() return outer end end",
    )
    .unwrap();
    let forward = &chunk.prototypes[0];
    let forwarded = &forward.upvalues[0];
    assert_eq!(forwarded.name.as_bytes(), b"outer");
    assert!(matches!(forwarded.source, UpvalueSource::ParentRegister(_)));
    let descendant = &forward.prototypes[0];
    assert_eq!(descendant.upvalues.len(), 1);
    assert_eq!(descendant.upvalues[0].name.as_bytes(), b"outer");
    assert_eq!(
        descendant.upvalues[0].source,
        UpvalueSource::ParentUpvalue(0)
    );

    let shadow = &chunk.prototypes[1];
    assert_eq!(
        shadow.upvalues,
        [] as [crate::bytecode::UpvalueDescriptor; 0]
    );
    let shadowed_descendant = &shadow.prototypes[0];
    assert_eq!(shadowed_descendant.upvalues.len(), 1);
    assert_eq!(shadowed_descendant.upvalues[0].name.as_bytes(), b"outer");
    assert!(matches!(
        shadowed_descendant.upvalues[0].source,
        UpvalueSource::ParentRegister(_)
    ));
}

#[test]
fn indexed_targets_are_captured_before_writes() {
    let vm = run("local t = {}; local i = 1; t[i], i = 7, 2; x, y = t[1], t[2]");
    assert_eq!(vm.runtime().get_global("x"), Value::Number(7.0));
    assert_eq!(vm.runtime().get_global("y"), Value::Nil);
}

#[test]
fn emits_wide_constant_and_global_references_after_the_compact_pool_limit() {
    let mut source = String::new();
    for value in 0..=u32::from(u16::MAX) {
        writeln!(source, "result = {value}").unwrap();
    }
    source.push_str("wide_result = result\n");
    let chunk = compile(&source).unwrap();
    assert!(chunk.constants.len() > usize::from(u16::MAX));
    assert!(
        chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::LoadConstantWide))
    );
    assert!(
        chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::SetGlobalWide))
    );
    assert_eq!(chunk.validate(), Ok(()));

    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&chunk).unwrap();
    assert_eq!(
        vm.runtime().get_global("wide_result"),
        Value::Number(f64::from(u16::MAX))
    );
}

#[test]
fn indexed_target_expressions_run_before_rhs_expressions() {
    let chunk = compile("local t = {}; t[f()] = g()").unwrap();
    let global_names: Vec<_> = chunk
        .constants
        .iter()
        .filter_map(|constant| match constant {
            Constant::String(name) => {
                Some(String::from_utf8(name.as_bytes().to_vec()).expect("global names are UTF-8"))
            }
            _ => None,
        })
        .collect();
    assert_eq!(global_names, ["f", "g"]);
}

#[test]
fn vararg_prototypes_keep_legacy_arg_for_fixed_and_pure_functions() {
    let chunk = compile(
        "function fixed(value, ...) return arg end; \
             function pure(...) return arg end; \
             function modern(...) return ... end",
    )
    .unwrap();
    let fixed = &chunk.prototypes[0];
    assert!(fixed.is_vararg);
    assert_eq!(fixed.parameter_count, 1);
    assert!(fixed.arg_register.is_some());
    let pure = &chunk.prototypes[1];
    assert!(pure.is_vararg);
    assert_eq!(pure.parameter_count, 0);
    assert!(pure.arg_register.is_some());
    let modern = &chunk.prototypes[2];
    assert!(modern.is_vararg);
    assert_eq!(modern.arg_register, None);
    assert!(
        modern
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::ReturnVararg))
    );
}

#[test]
fn varargs_adjust_forward_and_expand_in_calls_returns_and_tables() {
    let vm = run("function relay(prefix, ...) return prefix, ... end; \
             first, second, third = relay(1, 2, 3); \
             function values(...) local a, b, c = ...; return a, b, c end; \
             nil_a, nil_b, nil_c = values(4, nil, 6); \
             function adjusted(...) local a, b, c, d = 1, ...; return a, b, c, d end; \
             adjusted_a, adjusted_b, adjusted_c, adjusted_d = adjusted(2, 3); \
             function single(...) return (...), 99 end; \
             single_a, single_b, single_c = single(12, 13); \
             function triplet(a, b, c) call_a, call_b, call_c = a, b, c end; \
             function forward(...) triplet(1, ...) end; forward(2, 3); \
             function tabled(...) return {...} end; \
             packed = tabled(7, 8); table_a, table_b = packed[1], packed[2]; \
             function fixed(head, ...) return arg.n, arg[1], arg[2] end; \
             arg_n, arg_a, arg_b = fixed(0, 9, 10); \
             function pure(...) return arg end; pure_arg = pure(11, nil, 13); \
             function modern(...) local value = ...; return arg end; modern_arg = modern(12)");
    assert_eq!(vm.runtime().get_global("first"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("second"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("third"), Value::Number(3.0));
    assert_eq!(vm.runtime().get_global("nil_a"), Value::Number(4.0));
    assert_eq!(vm.runtime().get_global("nil_b"), Value::Nil);
    assert_eq!(vm.runtime().get_global("nil_c"), Value::Number(6.0));
    assert_eq!(vm.runtime().get_global("adjusted_a"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("adjusted_b"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("adjusted_c"), Value::Number(3.0));
    assert_eq!(vm.runtime().get_global("adjusted_d"), Value::Nil);
    assert_eq!(vm.runtime().get_global("single_a"), Value::Number(12.0));
    assert_eq!(vm.runtime().get_global("single_b"), Value::Number(99.0));
    assert_eq!(vm.runtime().get_global("single_c"), Value::Nil);
    assert_eq!(vm.runtime().get_global("call_a"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("call_b"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("call_c"), Value::Number(3.0));
    assert_eq!(vm.runtime().get_global("table_a"), Value::Number(7.0));
    assert_eq!(vm.runtime().get_global("table_b"), Value::Number(8.0));
    assert_eq!(vm.runtime().get_global("arg_n"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("arg_a"), Value::Number(9.0));
    assert_eq!(vm.runtime().get_global("arg_b"), Value::Number(10.0));
    let Value::Table(pure_arg) = vm.runtime().get_global("pure_arg") else {
        panic!("pure vararg function did not return a compatibility arg table");
    };
    assert_eq!(
        vm.runtime().table_get(pure_arg, &Value::from("n")).unwrap(),
        Value::Number(3.0)
    );
    assert_eq!(
        vm.runtime()
            .table_get(pure_arg, &Value::Number(1.0))
            .unwrap(),
        Value::Number(11.0)
    );
    assert_eq!(
        vm.runtime()
            .table_get(pure_arg, &Value::Number(2.0))
            .unwrap(),
        Value::Nil
    );
    assert_eq!(
        vm.runtime()
            .table_get(pure_arg, &Value::Number(3.0))
            .unwrap(),
        Value::Number(13.0)
    );
    assert_eq!(vm.runtime().get_global("modern_arg"), Value::Nil);
}

#[test]
fn final_call_results_expand_in_calls_returns_and_tables() {
    let vm = run("function produce() return 1, 2, 3 end; \
             function relay() return produce() end; \
             function pass(prefix, ...) return ... end; \
             a, b, c = pass(0, produce()); \
             d, e, f = relay(); \
             function tabled() return {0, produce()} end; \
             packed = tabled(); g, h, i, j = packed[1], packed[2], packed[3], packed[4]");
    assert_eq!(vm.runtime().get_global("a"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("b"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("c"), Value::Number(3.0));
    assert_eq!(vm.runtime().get_global("d"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("e"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("f"), Value::Number(3.0));
    assert_eq!(vm.runtime().get_global("g"), Value::Number(0.0));
    assert_eq!(vm.runtime().get_global("h"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("i"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("j"), Value::Number(3.0));
}

#[test]
fn anonymous_call_after_a_table_assignment_starts_a_new_statement() {
    let vm = run("a = {}\n(function (x) a=x end)(23)");
    assert_eq!(vm.runtime().get_global("a"), Value::Number(23.0));
}

#[test]
fn do_scope_does_not_leak() {
    let vm = run("do local x = 2; y = x + 1 end; z = x");
    assert_eq!(vm.runtime().get_global("y"), Value::Number(3.0));
    assert_eq!(vm.runtime().get_global("z"), Value::Nil);
}

#[test]
fn indexed_lhs_is_captured_before_rhs_calls() {
    let chunk =
        compile("local t = {}; i = 1; t[i] = rhs(); captured = t[1]; changed = t[9]").unwrap();
    let mut vm = Vm::new();
    vm.runtime_mut()
        .register_global_native("rhs", rhs_changes_i)
        .unwrap();
    vm.execute(&chunk).unwrap();
    assert_eq!(vm.runtime().get_global("captured"), Value::Number(42.0));
    assert_eq!(vm.runtime().get_global("changed"), Value::Nil);
}

#[test]
fn local_functions_stay_local_and_if_else_selects_a_branch() {
    let vm = run(
        "local function choose(value) if value then return 'yes' else return 'no' end end; \
             true_value = choose(true); false_value = choose(false); joined = 'a' .. 'b' .. 'c'",
    );
    assert_eq!(vm.runtime().get_global("choose"), Value::Nil);
    assert_eq!(vm.runtime().get_global("true_value"), Value::string("yes"));
    assert_eq!(vm.runtime().get_global("false_value"), Value::string("no"));
    assert_eq!(vm.runtime().get_global("joined"), Value::string("abc"));
}

#[test]
fn captures_mutate_the_ancestor_local() {
    let vm = run(
        "local outer = 7; local function increment() outer = outer + 1; return outer end; \
             first = increment(); second = increment()",
    );
    assert_eq!(vm.runtime().get_global("first"), Value::Number(8.0));
    assert_eq!(vm.runtime().get_global("second"), Value::Number(9.0));
}

#[test]
fn recursive_and_mutually_recursive_plain_functions_capture_local_bindings() {
    let vm = run("local one, even, odd; \
             function one(n) if n <= 0 then return 0 end return one(n - 1) end; \
             function even(n) if n <= 0 then return true end return odd(n - 1) end; \
             function odd(n) if n <= 0 then return false end return even(n - 1) end; \
             self_result = one(12); mutual_result = even(12)");
    assert_eq!(vm.runtime().get_global("self_result"), Value::Number(0.0));
    assert_eq!(
        vm.runtime().get_global("mutual_result"),
        Value::Boolean(true)
    );
}

#[test]
fn captures_survive_outer_return_and_forward_through_nested_functions() {
    let vm = run("function make(start) \
               local value = start; \
               local function middle() \
                 local function next() value = value + 1; return value end; \
                 return next \
               end; \
               return middle \
             end; \
             counter = make(4)(); first = counter(); second = counter()");
    assert_eq!(vm.runtime().get_global("first"), Value::Number(5.0));
    assert_eq!(vm.runtime().get_global("second"), Value::Number(6.0));
}

#[test]
fn table_constructor_results_remain_distinct_for_sibling_expressions() {
    let vm = run("first, second = { 1 }, { 2 }; distinct = first ~= second; \
             assert({ 1 } ~= { 2 }); \
             outer = { { 7 }, { 8 } }; nested_distinct = outer[1] ~= outer[2]");
    assert_eq!(vm.runtime().get_global("distinct"), Value::Boolean(true));
    assert_eq!(
        vm.runtime().get_global("nested_distinct"),
        Value::Boolean(true)
    );
}

#[test]
fn anonymous_function_closures_capture_ancestor_locals() {
    let vm =
        run("local base = 5; local add = function(value) return base + value end; result = add(7)");
    assert_eq!(vm.runtime().get_global("result"), Value::Number(12.0));
}

#[test]
fn generic_for_evaluates_setup_once_binds_results_and_stops_on_nil() {
    let chunk = compile(
        "sum = 0; for index, value in setup() do sum = sum + index + value end; leaked = index",
    )
    .unwrap();
    let mut vm = Vm::new();
    vm.runtime_mut()
        .register_global_native("iter", test_iterator)
        .unwrap();
    vm.runtime_mut()
        .register_global_native("setup", setup_iterator)
        .unwrap();
    vm.execute(&chunk).unwrap();
    assert_eq!(vm.runtime().get_global("setup_calls"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("sum"), Value::Number(66.0));
    assert_eq!(vm.runtime().get_global("leaked"), Value::Nil);
}

#[test]
fn loop_iterations_close_captured_locals_before_the_next_iteration() {
    let vm = run("local first, last; \
             for i = 1, 3 do \
               local function relay() return function() return i end end; \
               local closure = relay(); \
               if first then last = closure else first = closure end \
             end; \
             numeric_first, numeric_last = first(), last(); \
             local while_first, while_last; local n = 1; \
             while n <= 3 do \
               local local_n = n; \
               local function relay() return function() return local_n end end; \
               local closure = relay(); \
               if while_first then while_last = closure else while_first = closure end; \
               n = n + 1 \
             end; \
             while_first_value, while_last_value = while_first(), while_last()");
    assert_eq!(vm.runtime().get_global("numeric_first"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("numeric_last"), Value::Number(3.0));
    assert_eq!(
        vm.runtime().get_global("while_first_value"),
        Value::Number(1.0)
    );
    assert_eq!(
        vm.runtime().get_global("while_last_value"),
        Value::Number(3.0)
    );

    let chunk = compile(
        "local first, last; \
             for index in iter, nil, 0 do \
               local function relay() return function() return index end end; \
               local closure = relay(); \
               if first then last = closure else first = closure end \
             end; \
             generic_first, generic_last = first(), last()",
    )
    .unwrap();
    let mut vm = Vm::new();
    vm.runtime_mut()
        .register_global_native("iter", test_iterator)
        .unwrap();
    vm.execute(&chunk).unwrap();
    assert_eq!(vm.runtime().get_global("generic_first"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("generic_last"), Value::Number(3.0));
}

#[test]
fn while_bodies_close_captured_locals_that_reuse_condition_registers() {
    // A body statement boundary frees the condition's temporary, so the
    // first body local can occupy it. That local must still get a fresh
    // cell on every iteration.
    let vm = run("local closures = {}; local i = 0; \
             while i < 3 do \
               i = i + 1; \
               local assigned; assigned = i * 10; \
               local initialized = i * 100; \
               closures[i] = function() return assigned + initialized end \
             end; \
             first, last = closures[1](), closures[3]()");
    assert_eq!(vm.runtime().get_global("first"), Value::Number(110.0));
    assert_eq!(vm.runtime().get_global("last"), Value::Number(330.0));
}

#[test]
fn while_repeat_and_break_preserve_loop_and_repeat_scopes() {
    let vm = run(
        "local count = 0; while count < 10 do count = count + 1; if count == 3 then break end end; \
             while_result = count; \
             local n = 0; repeat local current = n; n = n + 1; repeat_value = current until current == 2; \
             repeat_after = current; \
             outer, inner = 0, 0; while outer < 3 do outer = outer + 1; repeat inner = inner + 1; break until false; if outer == 2 then break end end; \
             for i = 1, 10 do if i == 4 then break end numeric_sum = (numeric_sum or 0) + i end",
    );
    assert_eq!(vm.runtime().get_global("while_result"), Value::Number(3.0));
    assert_eq!(vm.runtime().get_global("repeat_value"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("repeat_after"), Value::Nil);
    assert_eq!(vm.runtime().get_global("outer"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("inner"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("numeric_sum"), Value::Number(6.0));
}

#[test]
fn elseif_selects_one_scoped_arm_and_keeps_break_and_return_control_flow() {
    let vm = run("local function choose(value) \
               if value == 1 then return 'one' \
               elseif value == 2 then return 'two' \
               elseif value == 3 then return 'three' \
               else return 'other' end \
             end; \
             first, second, third, fourth = choose(1), choose(2), choose(3), choose(4); \
             selected = nil; if false then local hidden = 1 elseif true then local visible = 2; selected = visible else selected = 3 end; leaked = visible; \
             total = 0; for i = 1, 5 do if i == 3 then break elseif i == 4 then total = 99 else total = total + i end end");
    assert_eq!(vm.runtime().get_global("first"), Value::string("one"));
    assert_eq!(vm.runtime().get_global("second"), Value::string("two"));
    assert_eq!(vm.runtime().get_global("third"), Value::string("three"));
    assert_eq!(vm.runtime().get_global("fourth"), Value::string("other"));
    assert_eq!(vm.runtime().get_global("selected"), Value::Number(2.0));
    assert_eq!(vm.runtime().get_global("leaked"), Value::Nil);
    assert_eq!(vm.runtime().get_global("total"), Value::Number(3.0));
}

/// The generic-`for` iterator call carries its kind and no stored name: the
/// VM spells `for iterator` only when an error or traceback asks, so
/// compiling a loop allocates no name string.
#[test]
fn generic_for_iterator_call_site_stores_no_name() {
    let chunk = compile_named(
        "loop.lua",
        b"for _ in next, {} do end",
        CompilerConfig::default(),
    )
    .unwrap();
    let sites = chunk.debug.call_sites();
    assert_eq!(sites.len(), 1);
    assert_eq!(sites[0].kind(), Some(CallKind::ForIterator));
    assert!(sites[0].is_iterator());
    assert!(sites[0].name().is_none());
}

#[test]
fn chunks_record_a_live_register_count_for_every_instruction() {
    let chunk = compile("local a = 1; local b = 2; return a + b").unwrap();
    let counts = chunk.live_register_counts();
    assert_eq!(counts.len(), chunk.code.len());
    // Straight line: nothing is live before the first local exists, and the
    // two locals stay live until the add reads them.
    let add = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::Add))
        .unwrap();
    assert!(counts[add] >= 2);
    assert!(counts[0] <= 1);
}

#[test]
fn live_register_counts_follow_loops_calls_and_nested_blocks() {
    // Loop: the control window and the accumulator stay live around the
    // latch, but the body's temporaries do not.
    let chunk = compile("local s = 0; for i = 1, 3 do s = s + i * 2 end; return s").unwrap();
    let latch = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForLoop))
        .unwrap();
    let control = usize::from(chunk.code[latch].a());
    assert_eq!(chunk.live_register_count(latch), control + 4);

    // Call: the callee and its argument are live at the call, and its
    // result register is dead until the call writes it.
    let chunk = compile("local f = print; local x = 1; f(x, x); return 0").unwrap();
    let call = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::Call))
        .unwrap();
    let base = usize::from(chunk.code[call].a());
    assert!(chunk.live_register_count(call) > base + 1);

    // Nested block: a local is live inside its block and a temporary above
    // it is not, so the count there equals the local's register plus one.
    let chunk = compile("do local inner = {}; inner.x = 1; end; return 1").unwrap();
    let local = chunk
        .debug
        .local_variables()
        .iter()
        .find(|local| local.name().as_bytes() == b"inner")
        .unwrap();
    let set_field = chunk
        .code
        .iter()
        .position(|instruction| {
            matches!(
                instruction.opcode(),
                Ok(OpCode::SetField | OpCode::SetTable)
            )
        })
        .unwrap();
    assert!(chunk.live_registers(set_field).contains(local.register()));
    let end = chunk.code.len() - 1;
    assert!(!chunk.live_registers(end).contains(local.register()));
}

#[test]
fn locals_filled_by_moves_start_after_the_moves() {
    // The call lands above the locals and moves fill them. While the call
    // runs, the locals' registers hold stale temporaries (here the table
    // passed to setmetatable), so they must be neither in scope nor live.
    let chunk = compile(
        "local mt = {}; setmetatable({}, mt); local a, b = pcall(collectgarbage); return a, b",
    )
    .unwrap();
    let call = chunk
        .code
        .iter()
        .rposition(|instruction| instruction.opcode() == Ok(OpCode::Call))
        .unwrap();
    // One move per local follows the call.
    assert!(
        chunk.code[call + 1..call + 3]
            .iter()
            .all(|instruction| instruction.opcode() == Ok(OpCode::Move))
    );
    for name in [b"a".as_slice(), b"b"] {
        let local = chunk
            .debug
            .local_variables()
            .iter()
            .find(|local| local.name().as_bytes() == name)
            .unwrap();
        assert_eq!(local.start_pc(), call + 3);
        assert!(!chunk.live_registers(call + 1).contains(local.register()));
    }
}

#[test]
fn dead_temporaries_are_not_cleared_after_statements() {
    let chunk = compile(
        "local t = {}; local n = 0; \
         for i = 1, 2000 do t[i] = i; n = n + #t end; return n",
    )
    .unwrap();
    let prep = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForPrep))
        .unwrap();
    let latch = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForLoop))
        .unwrap();
    for instruction in &chunk.code[prep + 1..latch] {
        assert!(!matches!(
            instruction.opcode(),
            Ok(OpCode::ClearRange | OpCode::LoadNil)
        ));
    }
}

#[test]
fn uninitialized_locals_start_nil_after_a_reused_temporary() {
    run("local a = {1, 2, 3}; local s = (a[1] + a[2]) .. 'x'; \
         for i = 1, 3 do local u; assert(u == nil); u = i end; \
         do local v; assert(v == nil) end; \
         local w; assert(w == nil)");
}

#[test]
fn objects_held_only_by_dead_temporaries_are_collected() {
    // Nothing clears the temporaries these tables were built in. Only the
    // live register counts keep them from being roots.
    run("local weak = setmetatable({}, {__mode = 'v'}); \
         local function id(value) return value end; \
         for i = 1, 3 do \
           weak[1] = {i}; \
           collectgarbage(); \
           assert(weak[1] == nil) \
         end; \
         weak[2] = id({}); \
         collectgarbage(); \
         assert(weak[2] == nil); \
         local n = #id({1, 2, 3}); \
         collectgarbage(); \
         weak[3] = {n}; weak[3] = nil; \
         assert(n == 3)");
}

#[test]
fn a_dumped_and_loaded_chunk_still_collects_dead_temporaries() {
    let source = "local n=0; local weak=setmetatable({[1]={}}, {__mode='kv'}); \
         local function f() \
           while weak[1] do \
             local garbage=n..n..n..n; n=n+1; \
             assert(n < 200000) \
           end \
         end; \
         f(); \
         assert(n > 0); \
         local keep = setmetatable({}, {__mode='v'}); \
         keep[1] = {}; keep[2] = ({1, 2})[1] + 1; \
         collectgarbage(); \
         assert(keep[1] == nil)";
    let compiled = compile(source).unwrap();
    let bytes = crate::bytecode::binary::dump(&compiled).unwrap();
    let loaded = crate::bytecode::binary::load(&bytes).unwrap();
    // Like compiled source, a loaded function is analysed on its first call.
    assert!(!loaded.has_live_registers());
    assert!(
        loaded
            .prototypes
            .iter()
            .all(|child| !child.has_live_registers())
    );
    assert_eq!(
        loaded.live_register_counts(),
        compiled.live_register_counts()
    );
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&loaded).unwrap();
    // The prototypes are shared with the closures that ran, which computed
    // the analysis on entry, before any collection rooted their frames.
    assert!(
        loaded
            .prototypes
            .iter()
            .all(|child| child.has_live_registers())
    );
}

#[test]
fn compiling_leaves_the_live_register_analysis_to_the_first_call() {
    let chunk = compile("local function f(x) return x end ; return f(1)").unwrap();
    assert!(!chunk.has_live_registers());
    assert!(!chunk.prototypes[0].has_live_registers());
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&chunk).unwrap();
    // The prototype is shared with the closure that ran, which computed it
    // on entry: the call path needs the registers live at pc 0, and a
    // collection roots frames by the sets.
    assert!(chunk.prototypes[0].has_live_registers());
}

#[test]
fn cloned_chunks_share_the_live_register_analysis() {
    let chunk = compile("local a = {} ; return a").unwrap();
    chunk.compute_live_registers();
    assert!(chunk.has_live_registers());
    let clone = chunk.clone();
    assert!(clone.has_live_registers());
    assert_eq!(clone.live_register_counts(), chunk.live_register_counts());
}

/// A generated chunk of `statements` statements with loops and branches.
fn generated_statements(statements: usize) -> String {
    let mut source = String::from("local t, n = {}, 0\n");
    for index in 0..statements {
        match index % 4 {
            0 => writeln!(source, "t[{}] = n + {}", index % 50, index).unwrap(),
            1 => writeln!(source, "n = n + (t[{}] or 0) * 2", index % 50).unwrap(),
            2 => writeln!(source, "if n > {index} then n = n - 1 end").unwrap(),
            _ => writeln!(source, "for i = 1, 2 do n = n + i end").unwrap(),
        }
    }
    source.push_str("return n\n");
    source
}

/// The liveness analysis must stay linear in the chunk length, so compiling
/// a 100k-statement chunk costs the same as before the analysis existed plus
/// a small constant factor. Measured against b4a48e1 through `load` in the
/// CLI: 0.486 s before, 0.517 s after (1.06x). The parser dominates this
/// size and has its own growth, so this test times the analysis by itself.
#[test]
fn compiling_a_hundred_thousand_statements_keeps_the_analysis_linear() {
    use std::time::{Duration, Instant};
    let small = compile(generated_statements(25_000)).unwrap();
    let large = compile(generated_statements(100_000)).unwrap();
    assert!(large.code.len() > 200_000);
    large.compute_live_registers();
    assert!(large.has_live_registers());
    assert_eq!(large.live_register_counts().len(), large.code.len());
    let analysis_time = |chunk: &Chunk| {
        let mut best = Duration::MAX;
        for _ in 0..3 {
            // A clone of an analysed chunk shares the analysis; rebuild one
            // that has to compute it.
            let fresh = Chunk {
                live: crate::bytecode::LiveRegisters::default(),
                ..chunk.clone()
            };
            let start = Instant::now();
            fresh.compute_live_registers();
            best = best.min(start.elapsed());
        }
        best
    };
    let small_time = analysis_time(&small);
    let large_time = analysis_time(&large);
    eprintln!("analysis 25k: {small_time:?}, 100k: {large_time:?}");
    // About 4x the work; anything near 16x would be quadratic.
    assert!(
        large_time < small_time * 8 + Duration::from_millis(5),
        "analysis time grew super-linearly: {small_time:?} -> {large_time:?}"
    );
}

/// Opcodes between the numeric-for prep and its latch.
fn loop_body_ops(source: &str) -> Vec<OpCode> {
    let chunk = compile(source).unwrap();
    let prep = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForPrep))
        .unwrap();
    let latch = chunk
        .code
        .iter()
        .position(|instruction| instruction.opcode() == Ok(OpCode::NumericForLoop))
        .unwrap();
    chunk.code[prep + 1..latch]
        .iter()
        .map(|instruction| instruction.opcode().unwrap())
        .collect()
}

fn run_returning(source: &str) -> Vec<Value> {
    let chunk = compile(source).unwrap();
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&chunk).unwrap()
}

#[test]
fn array_write_loop_uses_its_locals_in_place() {
    let ops = loop_body_ops("local p={} for i=1,2000 do p[i]=i end");
    assert_eq!(ops, vec![OpCode::SetTable]);
}

#[test]
fn table_field_arithmetic_and_comparison_operands_are_not_moved() {
    for source in [
        "local t,s={},0 for i=1,3 do s = t[i] end",
        "local t,s={},0 for i=1,3 do s = t.f end",
        "local t,i2,s={},1,0 for i=1,3 do t.f = i2 end",
        "local x,y,s=1,2,0 for i=1,3 do s = x*y + i end",
        "local x,y,s=1,2,false for i=1,3 do s = x<y end",
        "local x,y,s=1,2,0 for i=1,3 do if x<=y then s = 1 end end",
    ] {
        let ops = loop_body_ops(source);
        assert!(!ops.contains(&OpCode::Move), "{source}: {ops:?}");
    }
}

#[test]
fn multiple_assignment_conflicts_use_the_old_value() {
    let result = run_returning(
        "local a,i={},1 a[i], i = 10, 2 \
         local b,j={},1 j, b[j] = 3, 20 \
         local c,k={},1 c[k], k = k, 5 \
         return a[1], a[2], i, b[1], b[3], j, c[1], k",
    );
    assert_eq!(
        result,
        vec![
            Value::Number(10.0),
            Value::Nil,
            Value::Number(2.0),
            Value::Number(20.0),
            Value::Nil,
            Value::Number(3.0),
            Value::Number(1.0),
            Value::Number(5.0),
        ]
    );
    // Swaps read the old values.
    let result = run_returning("local x,y=1,2 x,y=y,x return x,y");
    assert_eq!(result, vec![Value::Number(2.0), Value::Number(1.0)]);
    let result = run_returning("local t={} local i=1 i,t[i]=i+1,'v' return i,t[1],t[2]");
    assert_eq!(
        result,
        vec![Value::Number(2.0), Value::from("v"), Value::Nil]
    );
}

#[test]
fn closure_assigned_locals_are_copied_before_later_code_runs() {
    let result = run_returning(
        "local t1,t2={},{} local t=t1 \
         local function swap() t=t2 return 1 end \
         t[swap()]='x' \
         local u,v=1,2 \
         local function bump() u=100 return 0 end \
         local a,b = u, bump() \
         return t1[1], t2[1], a, b",
    );
    assert_eq!(
        result,
        vec![
            Value::from("x"),
            Value::Nil,
            Value::Number(1.0),
            Value::Number(0.0)
        ]
    );
}

#[test]
fn indexing_metamethods_see_in_place_operands_in_order() {
    let result = run_returning(
        "local log='' \
         local m=setmetatable({}, {__newindex=function(_,k,v) log=log..k..v end}) \
         local k,v='a','1' \
         m[k]=v \
         local n=setmetatable({}, {__index=function(_,k) return k..'!' end}) \
         return log, n[k]",
    );
    assert_eq!(result, vec![Value::from("a1"), Value::from("a!")]);
}

#[test]
fn local_declarations_write_initializers_in_place() {
    let chunk = compile("local a,b,c=1,2,3 return a").unwrap();
    assert!(
        !chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::Move)),
        "{:?}",
        chunk.code
    );
    let result = run_returning("local x=5 local a,b=x,x+1 x=9 return a,b,x");
    assert_eq!(
        result,
        vec![Value::Number(5.0), Value::Number(6.0), Value::Number(9.0)]
    );
}

#[test]
fn call_arguments_copy_locals_into_the_call_window() {
    // Arguments go to consecutive registers above the callee, so each local
    // argument and the callee are copied, as stock Lua's MOVEs do. This
    // documents the current shape: three moves and the call.
    let ops = loop_body_ops("local f,x=print,1 for i=1,3 do f(x,i) end");
    assert_eq!(
        ops,
        vec![OpCode::Move, OpCode::Move, OpCode::Move, OpCode::Call]
    );
}

/// The opcodes of the main chunk, for shape checks.
fn opcodes_of(chunk: &Chunk) -> Vec<OpCode> {
    chunk
        .code
        .iter()
        .map(|instruction| instruction.opcode().unwrap())
        .collect()
}

/// `while true` loads and tests no constant: the loop is its body and the
/// jump back, which lands on the body's first instruction. Any other
/// condition keeps its test, `while 1` among them.
#[test]
fn while_true_compiles_to_its_body_and_a_jump_back() {
    for source in [
        "local n = 0 while true do n = n + 1 if n > 3 then break end end return n",
        "local n = 0 while (true) do n = n + 1 if n > 3 then break end end return n",
    ] {
        let chunk = compile(source).unwrap();
        let opcodes = opcodes_of(&chunk);
        let tests = opcodes
            .iter()
            .filter(|opcode| **opcode == OpCode::JumpIfFalse)
            .count();
        let with_test = opcodes_of(&compile(source.replace("true", "1")).unwrap());
        // The constant and its test are gone.
        assert_eq!(opcodes.len() + 2, with_test.len(), "{source}: {opcodes:?}");
        assert_eq!(
            tests + 1,
            with_test
                .iter()
                .filter(|opcode| **opcode == OpCode::JumpIfFalse)
                .count(),
            "{source}: {opcodes:?}"
        );
        let back: Vec<i64> = chunk
            .code
            .iter()
            .enumerate()
            .filter(|(_, word)| word.opcode().unwrap() == OpCode::Jump && word.jump_offset() < 0)
            .map(|(at, word)| i64::try_from(at).unwrap() + 1 + i64::from(word.jump_offset()))
            .collect();
        // One backward jump, to the body's first instruction.
        assert_eq!(back.len(), 1, "{source}: {opcodes:?}");
        let landing = usize::try_from(back[0]).unwrap();
        assert_ne!(
            opcodes[landing],
            OpCode::LoadConstant,
            "{source}: {opcodes:?}"
        );
        assert_eq!(chunk.validate(), Ok(()));
        let mut vm = Vm::new();
        assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(4.0)]);
    }
    let chunk = compile("local n = 0 while false do n = n + 1 end return n").unwrap();
    let mut vm = Vm::new();
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(0.0)]);
}

/// Loops on `while true` leave through `break`, `return`, `goto` and an
/// error, hold closures over their locals and nest, as with the test they
/// used to make.
#[test]
fn while_true_loops_run_as_before() {
    let source = r#"
        local out = {}
        local n = 0
        while true do
            n = n + 1
            local captured = n
            out[#out + 1] = function() return captured end
            if n == 3 then break end
        end
        local sum = 0
        for i = 1, #out do sum = sum + out[i]() end
        local function first_over(limit)
            local i = 0
            while true do
                i = i + 1
                while true do
                    if i * i > limit then return i end
                    break
                end
            end
        end
        local jumps = 0
        while true do
            jumps = jumps + 1
            if jumps < 5 then goto continue end
            do break end
            ::continue::
        end
        local ok = pcall(function()
            local k = 0
            while true do
                k = k + 1
                if k == 7 then error("stop") end
            end
        end)
        local co = coroutine.wrap(function()
            local v = 0
            while true do v = v + coroutine.yield(v) end
        end)
        co()
        local yielded = co(2) + co(3)
        return sum, first_over(50), jumps, ok, yielded
    "#;
    for profile in [CompatibilityProfile::Lua52, CompatibilityProfile::Lua54] {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
        assert_eq!(chunk.validate(), Ok(()));
        let mut vm = Vm::with_profile(profile);
        install_base_with_profile(vm.runtime_mut(), profile).unwrap();
        let results = vm.execute(&chunk).unwrap();
        let numbers: Vec<f64> = results
            .iter()
            .filter_map(|value| match value {
                Value::Number(number) => Some(*number),
                #[allow(clippy::cast_precision_loss)]
                Value::Integer(integer) => Some(*integer as f64),
                _ => None,
            })
            .collect();
        assert_eq!(numbers, vec![6.0, 8.0, 5.0, 7.0], "{profile:?}");
        assert_eq!(results[3], Value::Boolean(false), "{profile:?}");
    }
}

#[test]
fn generic_for_compiles_to_a_call_and_a_loop_test_after_the_body() {
    let chunk = compile("local t = {} for k, v in next, t do local x = k end").unwrap();
    let opcodes = opcodes_of(&chunk);
    let call = opcodes
        .iter()
        .position(|opcode| *opcode == OpCode::ForInCall)
        .expect("a generic for emits ForInCall");
    // The test follows the call and jumps back to the body, which starts
    // after the entry jump that skips to the call.
    assert_eq!(opcodes[call + 1], OpCode::ForInLoop);
    let call_word = chunk.code[call];
    assert_eq!(call_word.b(), 2, "two loop variables");
    assert_eq!(call_word.c(), 0);
    let test_word = chunk.code[call + 1];
    assert_eq!(test_word.a(), call_word.a());
    let back = i32::from(i16::from_ne_bytes(test_word.d().to_ne_bytes()));
    let body = usize::try_from(i64::try_from(call + 2).unwrap() + i64::from(back)).unwrap();
    assert_eq!(
        opcodes[body - 1],
        OpCode::Jump,
        "the entry jump precedes the body"
    );
    let entry = chunk.code[body - 1];
    let landing =
        usize::try_from(i64::try_from(body).unwrap() + i64::from(entry.jump_offset())).unwrap();
    assert_eq!(landing, call, "the entry jump lands on the call");
    // The old nil test and the copies of the call window are gone.
    assert!(!opcodes.contains(&OpCode::Equal));
    assert!(!opcodes.contains(&OpCode::JumpIfFalse));
    assert!(!opcodes.contains(&OpCode::Call));
    // The loop variables sit right after the control registers.
    let key = chunk
        .debug
        .local_variables()
        .iter()
        .find(|local| local.name().as_bytes() == b"k")
        .unwrap();
    assert_eq!(key.register(), call_word.a() + 3);
    // The window reaches a + 5 even for a one-variable loop.
    let one = compile("for k in next, {} do end").unwrap();
    let call = opcodes_of(&one)
        .iter()
        .position(|opcode| *opcode == OpCode::ForInCall)
        .unwrap();
    assert_eq!(one.code[call].b(), 1);
    assert!(usize::from(one.register_count) > usize::from(one.code[call].a()) + 5);
    assert_eq!(one.validate(), Ok(()));
}

#[test]
fn generic_for_variable_count_sets_the_result_count() {
    for count in [1_usize, 2, 3, 4, 9] {
        let names = (1..=count)
            .map(|index| format!("v{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let chunk = compile(format!("for {names} in next, {{}} do end")).unwrap();
        let call = chunk
            .code
            .iter()
            .find(|instruction| instruction.opcode() == Ok(OpCode::ForInCall))
            .unwrap();
        assert_eq!(usize::from(call.b()), count);
        let window = usize::from(call.a()) + 3 + count.max(3);
        assert!(
            usize::from(chunk.register_count) >= window,
            "{count} variables"
        );
        assert_eq!(chunk.validate(), Ok(()));
    }
}

#[test]
fn generic_for_breaks_and_captures_close_before_the_call() {
    let chunk = compile(
        "local fs = {} for i, v in ipairs({1, 2}) do fs[i] = function() return i end if v == 2 then break end end",
    )
    .unwrap();
    let opcodes = opcodes_of(&chunk);
    let call = opcodes
        .iter()
        .position(|opcode| *opcode == OpCode::ForInCall)
        .unwrap();
    // The captured loop variable is closed on the path into the next call,
    // and again after the loop for a break.
    assert_eq!(opcodes[call - 1], OpCode::CloseUpvalue);
    assert_eq!(opcodes[call + 2], OpCode::CloseUpvalue);
    assert_eq!(chunk.validate(), Ok(()));
}

#[test]
fn generic_for_exit_line_follows_the_profile() {
    for (profile, exit_jump) in [
        (CompatibilityProfile::Lua51, false),
        (CompatibilityProfile::Lua54, true),
        (CompatibilityProfile::Lua55, true),
    ] {
        let chunk = compile_with_config(
            "for k in next, {} do\n  local x = k\nend\nreturn 1",
            CompilerConfig::new(profile),
        )
        .unwrap();
        let opcodes = opcodes_of(&chunk);
        let test = opcodes
            .iter()
            .position(|opcode| *opcode == OpCode::ForInLoop)
            .unwrap();
        assert_eq!(opcodes[test + 1] == OpCode::Jump, exit_jump, "{profile}");
        if exit_jump {
            // The extra jump reports the line of the closing `end`.
            let lines = chunk.debug.line_info();
            assert_eq!(lines[test + 1], 3, "{profile}");
        }
    }
}

#[test]
fn tail_call_operands_are_computed_in_the_call_window() {
    // The function and each argument are written to their call slots by
    // the instructions that produce them, as for a `Call`: no copies.
    let chunk = compile(
        "local function f(a, b) return g(a - 1, b + a % 3) end\n\
         local function m(o, a) return o:go(a * 2, a + 1) end",
    )
    .unwrap();
    for prototype in &chunk.prototypes {
        let opcodes: Vec<_> = prototype
            .code
            .iter()
            .map(|instruction| instruction.opcode().unwrap())
            .collect();
        let tail = opcodes
            .iter()
            .position(|opcode| *opcode == OpCode::TailCall)
            .expect("the function ends in a tail call");
        let moves_before = opcodes[..tail]
            .iter()
            .filter(|opcode| **opcode == OpCode::Move)
            .count();
        // The method's receiver is the one copy a method call needs.
        let expected = usize::from(opcodes.contains(&OpCode::GetField));
        assert!(
            moves_before <= expected,
            "{opcodes:?} copies {moves_before} operands"
        );
    }
}

#[test]
fn tail_calls_compute_the_same_results_with_operands_in_place() {
    let vm = run(
        "local function sum(n, total) if n == 0 then return total end \
         return sum(n - 1, total + n % 7) end\n\
         local obj = {k = 3}\n\
         function obj:go(a, b) return self.k + a * 10 + b end\n\
         local function call(o, a) return o:go(a * 2, a + 1) end\n\
         first = sum(1000, 0)\n\
         second = call(obj, 5)",
    );
    let expected: i64 = (1..=1000).map(|n| n % 7).sum();
    assert_eq!(
        vm.runtime().get_global("first"),
        Value::Number(expected as f64)
    );
    assert_eq!(
        vm.runtime().get_global("second"),
        Value::Number(3.0 + 100.0 + 6.0)
    );
}

#[test]
fn a_rotating_assignment_copies_only_the_value_a_write_would_clobber() {
    // `a, b, c, d = b, c, d, a` reads each local before its own write except
    // `a`, whose target is written first: one copy, then one move per target.
    let chunk = compile("local a, b, c, d = 1, 2, 3, 4\na, b, c, d = b, c, d, a").unwrap();
    let moves = chunk
        .code
        .iter()
        .filter(|instruction| instruction.opcode() == Ok(OpCode::Move))
        .count();
    assert_eq!(moves, 5, "{:?}", chunk.code);
}

#[test]
fn parallel_assignments_of_locals_keep_their_meaning() {
    let vm = run("local a, b, c, d = 1, 2, 3, 4\n\
         a, b, c, d = b, c, d, a\n\
         r1 = a * 1000 + b * 100 + c * 10 + d\n\
         local x, y = 1, 2\n\
         x, y = y, x\n\
         r2 = x * 10 + y\n\
         local p, q, r = 1, 2, 3\n\
         p, q, r = q, p, p\n\
         r3 = p * 100 + q * 10 + r\n\
         local u, v = 5, 6\n\
         u, v = v, u + v\n\
         r4 = u * 100 + v\n\
         local t = {}\n\
         local k = 1\n\
         k, t[k] = 2, 'first'\n\
         r5 = k .. t[1] .. tostring(t[2])\n\
         local m, n = 1, 2\n\
         local function bump() m = m + 10 return 0 end\n\
         m, n = n, bump() + m\n\
         r6 = m * 100 + n\n\
         local s1, s2, s3 = 'a', 'b', 'c'\n\
         s1, s2, s3 = s3, s1, s2\n\
         r7 = s1 .. s2 .. s3\n\
         local e1, e2, e3 = 1, 2\n\
         e1, e2, e3 = e2, e3, e1\n\
         r8 = tostring(e1) .. tostring(e2) .. tostring(e3)\n");
    let get = |name: &str| vm.runtime().get_global(name);
    assert_eq!(get("r1"), Value::Number(2341.0));
    assert_eq!(get("r2"), Value::Number(21.0));
    assert_eq!(get("r3"), Value::Number(211.0));
    assert_eq!(get("r4"), Value::Number(611.0));
    assert_eq!(get("r5"), Value::String("2firstnil".into()));
    assert_eq!(get("r6"), Value::Number(211.0));
    assert_eq!(get("r7"), Value::String("cab".into()));
    assert_eq!(get("r8"), Value::String("2nil1".into()));
}
