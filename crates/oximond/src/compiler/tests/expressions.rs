use super::*;

#[test]
fn rejects_assignments_to_const_and_control_variables() {
    let config = CompilerConfig::new(CompatibilityProfile::Lua54);
    assert!(matches!(
        compile_with_config("local value<const> = 1; value = 2", config),
        Err(CompileError::AssignToReadOnly { .. })
    ));
    assert!(matches!(
        compile_with_config(
            "for index = 1, 2 do index = 3 end",
            CompilerConfig::new(CompatibilityProfile::Lua55),
        ),
        Err(CompileError::AssignToReadOnly { .. })
    ));
}

#[test]
fn sibling_upvalue_identity_calls_keep_both_argument_lists_live() {
    run_with_config(
        "local debug = require 'debug'; \
             local function make() \
               local shared; \
               return function() return shared end, function() return shared end \
             end; \
             local first, second = make(); \
             assert(debug.upvalueid(first, 1) == debug.upvalueid(second, 1))",
        CompilerConfig::new(CompatibilityProfile::Lua53),
    );
}

#[test]
fn unreachable_wrapped_coroutines_clear_weak_values_but_yielded_closures_survive() {
    run("local weak=setmetatable({}, {__mode='v'}); \
             local wrapped=coroutine.wrap(function() \
               local value=10; \
               local function increment() value=value+1; return value end; \
               while true do coroutine.yield(increment) end \
             end); \
             weak[1]=wrapped; local increment=wrapped(); assert(increment()==11); \
             wrapped=nil; collectgarbage(); assert(weak[1]==nil); \
             assert(increment()==12)");
}

/// The default profile is `LuaJIT` 2.1, which lets a coroutine yield from
/// inside a protected call, as Lua 5.2 and later do. (Yield across a
/// metamethod is allowed by stock too and is still refused here; it is
/// tracked separately, so this test does not pin either answer.)
#[test]
fn yield_crosses_protected_call_boundaries() {
    run("local co=coroutine.create(function() \
               local ok,message=pcall(coroutine.yield,1); \
               assert(ok and message=='r'); return 9 \
             end); \
             local ok,value=coroutine.resume(co); \
             assert(ok and value==1 and coroutine.status(co)=='suspended'); \
             ok,value=coroutine.resume(co,'r'); \
             assert(ok and value==9 and coroutine.status(co)=='dead')");
}

#[test]
fn single_index_assignment_does_not_copy_its_result() {
    let chunk = compile("local values={}; for i=1,10 do values[i]=i*2 end").unwrap();
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
    let move_count = chunk.code[prep + 1..loop_pc]
        .iter()
        .filter(|instruction| instruction.opcode() == Ok(OpCode::Move))
        .count();

    assert_eq!(move_count, 0);
}

#[test]
fn table_loop_emits_no_dead_temporary_clears() {
    let chunk =
        compile("local values={}; for i=1,10 do values[i]=i*2 end; return values[4]").unwrap();
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
    assert!(chunk.code[prep + 1..loop_pc].iter().all(|instruction| {
        !matches!(
            instruction.opcode(),
            Ok(OpCode::ClearRange | OpCode::LoadNil)
        )
    }));

    let mut vm = Vm::new();
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::Number(8.0)]);
}

#[test]
fn compiler_config_applies_lua51_ambiguous_call_newline_rule() {
    let source = "value = target\n(3)";
    for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
        let error = compile_with_config(source, CompilerConfig::new(profile)).unwrap_err();
        let CompileError::Syntax(error) = error else {
            panic!("ambiguous call must be a syntax error");
        };
        assert_eq!(error.span.start, source.rfind('(').unwrap());
    }
    assert!(compile_with_config(source, CompilerConfig::new(CompatibilityProfile::Lua52)).is_ok());
    assert!(
        compile(source).is_err(),
        "default compilation is LuaJIT 2.1"
    );
}

#[test]
fn named_compilation_records_static_call_origins() {
    let chunk = compile_named(
        "debug-calls.lua",
        b"global(); local local_fn = global; local_fn(); \
              local object = {}; object.field = global; object.field(); \
              object:method(); (local_fn)(); (function() end)()",
        CompilerConfig::default(),
    )
    .unwrap();
    assert_eq!(chunk.debug.source_name().as_bytes(), b"debug-calls.lua");

    let call_sites: Vec<_> = chunk
        .debug
        .call_sites()
        .iter()
        .map(|site| {
            (
                site.name().map(|name| name.as_bytes().to_vec()),
                site.kind(),
            )
        })
        .collect();
    assert_eq!(
        call_sites,
        vec![
            (Some(b"global".to_vec()), Some(CallKind::Global)),
            (Some(b"local_fn".to_vec()), Some(CallKind::Local)),
            (Some(b"field".to_vec()), Some(CallKind::Field)),
            (Some(b"method".to_vec()), Some(CallKind::Method)),
            (Some(b"local_fn".to_vec()), Some(CallKind::Local)),
            (None, None),
        ]
    );
    for call_site in chunk.debug.call_sites() {
        assert!(matches!(
            chunk.code[call_site.pc()].opcode(),
            Ok(OpCode::Call
                | OpCode::CallOne
                | OpCode::CallVararg
                | OpCode::CallOpen
                | OpCode::TailCall
                | OpCode::TailCallVararg
                | OpCode::TailCallOpen)
        ));
    }
}

#[test]
fn one_argument_one_result_calls_use_call_one() {
    let chunk =
        compile("local function identity(value) return value end; local result = identity(7)")
            .unwrap();
    let call = chunk
        .code
        .iter()
        .copied()
        .find(|instruction| instruction.opcode() == Ok(OpCode::CallOne))
        .expect("fixed one-argument call should use CallOne");
    assert_ne!(call.a(), call.b());
    assert_ne!(call.a(), call.c());
    assert_eq!(chunk.validate(), Ok(()));
}

#[test]
fn records_call_instruction_source_lines() {
    let chunk = compile_named(
        "lines.lua",
        b"first()\nsecond()\n",
        CompilerConfig::default(),
    )
    .unwrap();
    assert_eq!(chunk.debug.line_info().len(), chunk.code.len());
    let call_lines: Vec<_> = chunk
        .debug
        .call_sites()
        .iter()
        .map(|site| chunk.debug.line_at(site.pc()))
        .collect();
    assert_eq!(call_lines, vec![Some(1), Some(2)]);
}

#[test]
fn records_lua_function_definition_bounds() {
    let chunk = compile_named(
        "definition-lines.lua",
        b"\nlocal function outer()\n  return function()\n    return 1\n  end\nend\n",
        CompilerConfig::default(),
    )
    .unwrap();
    let outer = &chunk.prototypes[0];
    let inner = &outer.prototypes[0];
    assert_eq!(outer.debug.definition_line(), 2);
    assert_eq!(outer.debug.last_definition_line(), 6);
    assert_eq!(inner.debug.definition_line(), 3);
    assert_eq!(inner.debug.last_definition_line(), 5);
}

#[test]
fn flattens_unparenthesized_concat_chains_into_one_register_range() {
    let chunk = compile_named(
        "concat.lua",
        b"return a .. b .. c .. d",
        CompilerConfig::default(),
    )
    .unwrap();
    let concatenations: Vec<_> = chunk
        .code
        .iter()
        .copied()
        .filter(|instruction| instruction.opcode() == Ok(OpCode::Concat))
        .collect();
    assert_eq!(concatenations.len(), 1);
    let instruction = concatenations[0];
    assert_eq!(instruction.c() - instruction.b() + 1, 4);
}

#[test]
fn nested_prototypes_share_source_name_and_record_upvalue_calls() {
    let chunk = compile_named(
        "nested-debug.lua",
        b"local target; local function outer() return function() return target() end end",
        CompilerConfig::default(),
    )
    .unwrap();
    let outer = &chunk.prototypes[0];
    let inner = &outer.prototypes[0];
    for prototype in [&chunk, outer, inner] {
        assert_eq!(
            prototype.debug.source_name().as_bytes(),
            b"nested-debug.lua"
        );
    }
    assert_eq!(inner.debug.call_sites().len(), 1);
    let call_site = &inner.debug.call_sites()[0];
    assert_eq!(call_site.name().unwrap().as_bytes(), b"target");
    assert_eq!(call_site.kind(), Some(CallKind::Upvalue));
}

#[test]
fn sibling_closures_capture_the_same_parent_binding() {
    let chunk = compile(
        "local shared; local function left() return shared end; \
             local function right() return shared end",
    )
    .unwrap();
    let left = &chunk.prototypes[0].upvalues[0];
    let right = &chunk.prototypes[1].upvalues[0];
    assert_eq!(left.name.as_bytes(), b"shared");
    assert_eq!(right.name.as_bytes(), b"shared");
    assert_eq!(left.source, right.source);
    assert!(matches!(left.source, UpvalueSource::ParentRegister(_)));
}

#[test]
fn pads_and_assigns_many_values() {
    let vm = run("local a, b, c = 1, 2; x, y = a, b, c");
    assert_eq!(vm.runtime().get_global("x"), Value::Number(1.0));
    assert_eq!(vm.runtime().get_global("y"), Value::Number(2.0));
}

#[test]
fn compile_bytes_preserves_non_utf8_short_and_long_string_literals() {
    let chunk = compile(b"short = '\xff'; long = [[\x80]]").unwrap();
    assert!(chunk.constants.iter().any(|constant| {
        matches!(constant, Constant::String(value) if value.as_bytes() == [255])
    }));
    assert!(chunk.constants.iter().any(|constant| {
        matches!(constant, Constant::String(value) if value.as_bytes() == [128])
    }));
}

#[test]
fn call_results_expand_only_when_final() {
    let vm = run("local a, b = assert(true, 9); x, y = a, b");
    assert_eq!(vm.runtime().get_global("x"), Value::Boolean(true));
    assert_eq!(vm.runtime().get_global("y"), Value::Number(9.0));
}

#[test]
fn method_call_results_expand_when_final_in_an_assignment() {
    let vm = run(
        "local object={}; function object:values() return nil, 'detail', 7 end; \
             local a,b,c=object:values(); first=a; second=b; third=c",
    );
    assert_eq!(vm.runtime().get_global("first"), Value::Nil);
    assert_eq!(vm.runtime().get_global("second"), Value::from("detail"));
    assert_eq!(vm.runtime().get_global("third"), Value::Number(7.0));
}

#[test]
fn runs_the_pinned_luajit_assignment_program() {
    let source = r"
local assert = assert
do --- local
  local a, b, c
  a, b, c = 0, 1
  assert(a == 0)
  assert(b == 1)
  assert(c == nil)
  a, b = a+1, b+1, a+b
  assert(a == 1)
  assert(b == 2)
  a, b, c = 0
  assert(a == 0)
  assert(b == nil)
  assert(c == nil)
end
do --- global !private_G
  a, b, c = 0, 1
  assert(a == 0)
  assert(b == 1)
  assert(c == nil)
  a, b = a+1, b+1, a+b
  assert(a == 1)
  assert(b == 2)
  a, b, c = 0
  assert(a == 0)
  assert(b == nil)
  assert(c == nil)
end
do --- local lhs in key on lhs
  local a = {}
  local i = 3
  i, a[i] = i+1, 20
  assert(i == 4)
  assert(a[3] == 20)
end
do --- global lhs in key on lhs !private_G
  a = {}
  i = 3
  i, a[i] = i+1, 20
  assert(i == 4)
  assert(a[3] == 20)
end
";
    let chunk = compile(source).unwrap();
    let mut vm = Vm::new();
    install_base(vm.runtime_mut()).unwrap();
    vm.execute(&chunk).unwrap();
}

// Native callbacks use a fallible interface even when a test callback
// cannot fail.

#[test]
fn compiles_not_and_table_length() {
    let chunk = compile("truth = not nil; size = #{}").unwrap();
    assert!(
        chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::Not))
    );
    assert!(
        chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::Length))
    );
}

#[test]
fn table_constructor_keeps_field_order_and_list_indexes() {
    let vm = run("local t = {10; label = 20, [3] = 30, 40}; \
             first, second, named, keyed = t[1], t[2], t.label, t[3]");
    assert_eq!(vm.runtime().get_global("first"), Value::Number(10.0));
    assert_eq!(vm.runtime().get_global("second"), Value::Number(40.0));
    assert_eq!(vm.runtime().get_global("named"), Value::Number(20.0));
    assert_eq!(vm.runtime().get_global("keyed"), Value::Number(30.0));
}

#[test]
fn large_and_nested_table_constructors_reclaim_field_temporaries() {
    let mut source = String::from("local t = {");
    for value in 1..=1_000 {
        if value > 1 {
            source.push(',');
        }
        source.push_str(&value.to_string());
    }
    source.push_str("}; result = t[1000]");
    let vm = run(&source);
    assert_eq!(vm.runtime().get_global("result"), Value::Number(1_000.0));

    let vm = run("nested = { { 1 }, { 2 }, { { 3 } } }; result = nested[3][1][1]");
    assert_eq!(vm.runtime().get_global("result"), Value::Number(3.0));
}

#[test]
fn power_uses_lua_precedence_and_accepts_unary_exponents() {
    let vm = run("left = -2^2; right = 2^-2; nested = 2^3^2");
    assert_eq!(vm.runtime().get_global("left"), Value::Number(-4.0));
    assert_eq!(vm.runtime().get_global("right"), Value::Number(0.25));
    assert_eq!(vm.runtime().get_global("nested"), Value::Number(512.0));
}

#[test]
fn deep_function_paths_and_methods_assign_to_the_final_field() {
    let vm = run("local a = {}; a.b = {}; a.b.c = {}; \
             function a.b.c.increment(value) return value + 1 end; \
             function a.b.c:apply(value) return self.increment(value) end; \
             dotted = a.b.c.increment(4); method = a.b.c:apply(8)");
    assert_eq!(vm.runtime().get_global("dotted"), Value::Number(5.0));
    assert_eq!(vm.runtime().get_global("method"), Value::Number(9.0));
}

/// Each call operand below is a local that a later operand reassigns
/// through a closure. Lua reads the function and each argument, and copies
/// a method's receiver, before evaluating what follows.
const LATE_REASSIGNED_CALL_OPERANDS: &str = r"
    local out = {}
    local object = { name = 'first' }
    function object:get(extra, more) return self.name .. extra .. (more or '') end
    local other = { name = 'second', get = object.get }
    local receiver = object
    out[#out + 1] = receiver:get((function() receiver = other return '!' end)())
    receiver = object
    out[#out + 1] = receiver:get('a', (function() receiver = other return 'b' end)())
    receiver = object
    local function steal() receiver = other return 'c' end
    out[#out + 1] = receiver:get(steal())
    local function tail(...)
        local local_receiver = object
        local function steal_local() local_receiver = other return 'd' end
        if ... then return local_receiver:get(steal_local(), ...) end
        return local_receiver:get((steal_local()))
    end
    out[#out + 1] = tail()
    out[#out + 1] = tail('e')
    local lookup = setmetatable({ name = 'meta' }, {})
    local looked_up = lookup
    getmetatable(lookup).__index = function()
        looked_up = other
        return function(self) return self.name end
    end
    out[#out + 1] = looked_up:get()
    local x = 'old'
    local function bump() x = 'new' return 'g' end
    local function join(a, b) return a .. b end
    out[#out + 1] = join(x, bump())
    x = 'old'
    out[#out + 1] = join(x, (bump()))
    x = 'old'
    out[#out + 1] = join(x, 'h', bump())
    local function tail_join() local y = 'old' local function set() y = 'new' return 'i' end return join(y, set()) end
    out[#out + 1] = tail_join()
    local callee = function(a) return 'first' .. tostring(a) end
    local function swap() callee = function() return 'second' end return 1 end
    out[#out + 1] = callee(swap())
    callee = function(a) return 'first' .. tostring(a) end
    out[#out + 1] = callee((swap()))
    result = out[1]
    for index = 2, #out do result = result .. ' ' .. out[index] end
";
const LATE_REASSIGNED_CALL_RESULT: &str =
    "first! firstab firstc firstd firstde meta oldg oldg oldh oldi first1 first1";

#[test]
fn call_operands_are_read_before_later_operands_reassign_them() {
    let vm = run_with_config(
        LATE_REASSIGNED_CALL_OPERANDS,
        CompilerConfig::new(CompatibilityProfile::Lua51),
    );
    assert_eq!(
        vm.runtime().get_global("result"),
        Value::from(LATE_REASSIGNED_CALL_RESULT)
    );
}

#[test]
fn compact_call_operands_are_read_before_later_operands_reassign_them() {
    // The nested call keeps more than 255 registers live under the normal
    // allocator, so the whole chunk is compiled with compact registers.
    let numbers = (0..110)
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "local wide = select('#', {numbers}, select('#', {numbers}))\n{LATE_REASSIGNED_CALL_OPERANDS}"
    );
    let config = CompilerConfig::new(CompatibilityProfile::Lua51);
    let parsed = syntax::parse_bytes_with_config(
        source.as_bytes(),
        syntax::ParserConfig::new(config.compatibility_profile()),
    )
    .unwrap();
    assert!(matches!(
        compile_main(LuaString::from("=?"), &parsed, config, false),
        Err(CompileError::RegisterExhausted { .. })
    ));
    let vm = run_with_config(&source, CompilerConfig::new(CompatibilityProfile::Lua51));
    assert_eq!(
        vm.runtime().get_global("result"),
        Value::from(LATE_REASSIGNED_CALL_RESULT)
    );
}

#[test]
fn call_operands_that_no_closure_assigns_are_not_copied() {
    // Only the method lookup copies the receiver, to the self slot.
    let chunk = compile("local o = {}; local x = 1; o:m(x, #o, g())").unwrap();
    let moves_from_receiver = chunk
        .code
        .iter()
        .filter(|instruction| instruction.opcode() == Ok(OpCode::Move) && instruction.b() == 0)
        .count();
    assert_eq!(moves_from_receiver, 0);
    let lookups: Vec<_> = chunk
        .code
        .iter()
        .filter(|instruction| instruction.opcode() == Ok(OpCode::SelfField))
        .collect();
    assert_eq!(lookups.len(), 1);
    assert_eq!(lookups[0].b(), 0, "the lookup reads the local itself");
    let chunk =
        compile("local o = {}; local x = 1; local function f() o = nil end; o:m(x, #o, g())")
            .unwrap();
    let get_method = chunk
        .code
        .iter()
        .position(|instruction| {
            matches!(
                instruction.opcode(),
                Ok(OpCode::GetField | OpCode::SelfField)
            )
        })
        .unwrap();
    assert!(
        chunk.code[..get_method]
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::Move) && instruction.b() == 0),
        "a receiver that a closure assigns is copied before the lookup"
    );
}

#[test]
fn a_method_call_reads_the_method_and_copies_the_receiver_in_one_instruction() {
    for source in [
        "local o = {}; return o:m(1, 2)",
        "local o = {}; return o.p.q:m(1)",
        "local o = {}; return o:m(1, 2, ...)",
        "local o = {}; return o:m(g())",
        "local o = {}; local r = o:m(1, 2); return r",
        "local o = {}; local r = o:m(1, 2, ...); return r",
        "local o = {}; local r = o:m(g()); return r",
    ] {
        let chunk =
            compile_with_config(source, CompilerConfig::new(CompatibilityProfile::Lua54)).unwrap();
        let opcodes: Vec<_> = chunk
            .code
            .iter()
            .map(|instruction| instruction.opcode().unwrap())
            .collect();
        let lookup = opcodes
            .iter()
            .position(|opcode| *opcode == OpCode::SelfField)
            .unwrap_or_else(|| panic!("{source}: {opcodes:?}"));
        // The method and the receiver stand in adjacent registers, as the
        // call needs them.
        let instruction = chunk.code[lookup];
        let call = chunk
            .code
            .iter()
            .rev()
            .find(|instruction| {
                matches!(
                    instruction.opcode(),
                    Ok(OpCode::Call
                        | OpCode::CallVararg
                        | OpCode::CallOpen
                        | OpCode::TailCall
                        | OpCode::TailCallVararg
                        | OpCode::TailCallOpen)
                )
            })
            .unwrap();
        assert_eq!(instruction.a(), call.a(), "{source}");
        assert!(
            !chunk.code[lookup..]
                .iter()
                .any(|later| later.opcode() == Ok(OpCode::Move)
                    && later.a() == instruction.a() + 1),
            "{source}: the receiver is copied once, by the lookup: {opcodes:?}"
        );
    }
}

#[test]
fn the_register_saving_allocator_also_fuses_the_method_lookup_and_the_receiver_copy() {
    // `compact_registers` is the fallback for a function that runs out of
    // registers otherwise; it places a method call's window by itself.
    let source = "local o = {k = 5}\n\
                  function o:m(a, b) return self.k + a + b end\n\
                  local function tail(x) return o:m(x, x) end\n\
                  return o:m(1, 2), (o:m(3, 4)), tail(10)";
    let config = CompilerConfig::new(CompatibilityProfile::Lua54);
    let parsed = syntax::parse_bytes_with_config(
        source.as_bytes(),
        syntax::ParserConfig::new(config.compatibility_profile()),
    )
    .unwrap();
    let chunk = compile_main(LuaString::from("=?"), &parsed, config, true).unwrap();
    assert!(
        chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::SelfField))
    );
    let mut vm = Vm::with_profile(config.compatibility_profile());
    install_base_with_profile(vm.runtime_mut(), config.compatibility_profile()).unwrap();
    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::Integer(8), Value::Integer(12), Value::Integer(25)]
    );
}

#[test]
fn a_method_call_on_a_key_too_large_for_a_field_instruction_still_copies_the_receiver() {
    // More than 255 distinct names push the method name out of the field
    // constants: the lookup is a `GetTable` and the receiver a `Move`.
    let mut source = String::from(
        "local o = {}
",
    );
    for index in 0..300 {
        let _ = writeln!(source, "o.f{index} = {index}");
    }
    source.push_str("return o:late(1)");
    let chunk = compile(&source).unwrap();
    assert!(
        chunk
            .code
            .iter()
            .any(|instruction| instruction.opcode() == Ok(OpCode::GetTable)),
        "the method name needs a register"
    );
}

#[test]
fn multiline_operators_and_method_lookup_have_instruction_lines() {
    for (source, opcode, line) in [
        ("return\n {}\n <\n 1", OpCode::LessThan, 4),
        ("if\n {}\n <\n 1 then end", OpCode::LessThan, 4),
        ("return\n {}\n ..\n 1", OpCode::Concat, 3),
        ("return\n {}\n ..\n 1\n ..\n 2", OpCode::Concat, 5),
        ("return\n (nil)\n :foo()", OpCode::SelfField, 3),
    ] {
        let chunk =
            compile_with_config(source, CompilerConfig::new(CompatibilityProfile::Lua54)).unwrap();
        let pc = chunk
            .code
            .iter()
            .position(|instruction| instruction.opcode() == Ok(opcode))
            .unwrap();
        assert_eq!(chunk.debug.line_at(pc), Some(line), "{source}");
    }
}

#[test]
fn nested_function_enforces_profile_upvalue_boundaries() {
    for profile in [
        CompatibilityProfile::Lua51,
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        let limit = if profile.is_lua51_family() { 60 } else { 255 };
        for count in [limit, limit + 1] {
            let mut source = String::new();
            for index in 0..count.min(130) {
                let _ = writeln!(source, "local v{index} = {index}");
            }
            source.push_str("return function()\n");
            for index in 130..count {
                let _ = writeln!(source, "local v{index} = {index}");
            }
            source.push_str("return function()\nlocal sum = 0\n");
            for index in 0..count {
                let _ = writeln!(source, "sum = sum + v{index}");
            }
            source.push_str("return sum\nend\nend");
            let result = compile_with_config(&source, CompilerConfig::new(profile));
            if count == limit {
                assert_eq!(
                    result.unwrap().prototypes[0].prototypes[0].upvalues.len(),
                    count
                );
            } else {
                assert!(
                    matches!(result, Err(CompileError::UpvalueExhausted { .. })),
                    "{profile} accepted more than {limit} upvalues"
                );
            }
        }
    }
}

#[test]
fn lexical_environment_counts_toward_the_255_upvalue_limit() {
    for profile in [
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ] {
        for environment_first in [false, true] {
            for count in [254, 255] {
                let mut source = String::new();
                for index in 0..130 {
                    let _ = writeln!(source, "local v{index} = {index}");
                }
                source.push_str("return function()\n");
                for index in 130..count {
                    let _ = writeln!(source, "local v{index} = {index}");
                }
                source.push_str("return function()\nlocal sum = 0\n");
                if environment_first {
                    source.push_str("sum = sum + global_value\n");
                }
                for index in 0..count {
                    let _ = writeln!(source, "sum = sum + v{index}");
                }
                if !environment_first {
                    source.push_str("sum = sum + global_value\n");
                }
                source.push_str("return sum\nend\nend");
                let result = compile_with_config(&source, CompilerConfig::new(profile));
                if count == 254 {
                    let chunk = result.unwrap();
                    let captures = &chunk.prototypes[0].prototypes[0].upvalues;
                    assert_eq!(captures.len(), 255);
                    assert_eq!(
                        captures
                            .iter()
                            .filter(|upvalue| upvalue.name.as_bytes() == b"_ENV")
                            .count(),
                        1
                    );
                } else {
                    assert!(
                        matches!(result, Err(CompileError::UpvalueExhausted { .. })),
                        "{profile} accepted 255 locals plus _ENV"
                    );
                }
            }
        }
    }
}

#[test]
fn constant_string_keys_compile_to_field_instructions() {
    let chunk = compile(
        "local t = {name = 1}; t.size = t.name; t:method(); function t.f() end; return t['x']",
    )
    .unwrap();
    let opcodes: Vec<_> = chunk
        .code
        .iter()
        .filter_map(|instruction| instruction.opcode().ok())
        .collect();
    assert_eq!(
        opcodes
            .iter()
            .filter(|opcode| **opcode == OpCode::SetField)
            .count(),
        3
    );
    assert_eq!(
        opcodes
            .iter()
            .filter(|opcode| **opcode == OpCode::GetField)
            .count(),
        3
    );
    assert!(!opcodes.contains(&OpCode::GetTable));
    assert!(!opcodes.contains(&OpCode::SetTable));
}

#[test]
fn field_names_past_the_byte_operand_load_their_key() {
    // Constants 0..=299 are numbers, so the field name's index needs more
    // than a byte and the access keeps a register key.
    let mut source = String::from("local sink = {");
    for value in 0..300 {
        let _ = write!(source, "{value}.5, ");
    }
    source.push_str("} return sink.late");
    let chunk = compile(&source).unwrap();
    let opcodes: Vec<_> = chunk
        .code
        .iter()
        .filter_map(|instruction| instruction.opcode().ok())
        .collect();
    assert!(opcodes.contains(&OpCode::GetTable));
    assert!(!opcodes.contains(&OpCode::GetField));
}

#[test]
fn subtracting_integer_zero_is_an_addition_only_in_lua54_and_later() {
    let has_equal = |profile, source| {
        let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
        chunk
            .code
            .iter()
            .filter_map(|instruction| instruction.opcode().ok())
            .any(|opcode| opcode == OpCode::Equal)
    };
    for source in [
        "local x = ... return x - 0",
        "local x = ... return x - (1 - 1)",
        "local x = ... return x - -0",
    ] {
        for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
            assert!(has_equal(profile, source), "{profile} {source}");
        }
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua53,
            CompatibilityProfile::LuaJit21,
        ] {
            assert!(!has_equal(profile, source), "{profile} {source}");
        }
    }
    // A nonzero constant, a float zero and a variable subtract as written.
    for source in [
        "local x = ... return x - 1",
        "local x = ... return x - 0.0",
        "local x, k = ... return x - k",
        "local x = ... return 0 - x",
    ] {
        assert!(!has_equal(CompatibilityProfile::Lua54, source), "{source}");
    }
}

#[test]
fn subtracting_integer_zero_keeps_the_sign_rule_of_addi() {
    let vm = run_with_config(
        "local z = -0.0; \
         assert(1 / (z - 0) == math.huge); \
         assert(1 / (z - 0.0) == -math.huge); \
         local k = 0; assert(1 / (z - k) == -math.huge); \
         assert(math.type(5 - 0) == 'integer'); \
         local t = setmetatable({}, {__sub = function() return 'sub' end, \
                                     __add = function() return 'add' end}); \
         assert(t - 0 == 'sub'); \
         assert(1 / ('-0.0' - 0) == -math.huge)",
        CompilerConfig::new(CompatibilityProfile::Lua54),
    );
    drop(vm);
}

#[test]
fn subtracting_a_folded_constant_zero_is_an_addition() {
    fn any_equal(chunk: &crate::bytecode::Chunk) -> bool {
        chunk
            .code
            .iter()
            .filter_map(|instruction| instruction.opcode().ok())
            .any(|opcode| opcode == OpCode::Equal)
            || chunk.prototypes.iter().any(|child| any_equal(child))
    }
    // A profile that rejects `<const>` does not compile the source at all.
    let has_equal = |profile, source: &str| {
        compile_with_config(source, CompilerConfig::new(profile))
            .is_ok_and(|chunk| any_equal(&chunk))
    };
    for source in [
        "local z = ... local k <const> = 0 return z - k",
        "local z = ... local k <const> = 0 return z - (k)",
        "local z = ... local k <const> = 0 return z - (k + 0)",
        "local z = ... local q <const> = 1 - 1 return z - q",
        "local z = ... return z - (2 // 2 - 1)",
        "local z = ... return z - (5 % 5)",
        "local z = ... return z - (0 | 0)",
        "local z = ... return z - (3 & 4)",
        "local z = ... return z - (5 ~ 5)",
        "local z = ... return z - ~-1",
        "local z = ... return z - (1 << 64)",
        "local z = ... return z - (-1 >> 64)",
        "local z = ... local k <const> = 0 return function() return z - k end",
        "local z = ... local a <const>, b <const> = 5, 0 return z - b",
    ] {
        for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
            assert!(has_equal(profile, source), "{profile} {source}");
        }
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::LuaJit21] {
            assert!(!has_equal(profile, source), "{profile} {source}");
        }
    }
    // Not constants: a plain local, a shadowed or redeclared name, any but the
    // last variable, a float result and a division by zero.
    for source in [
        "local z = ... local k = 0 return z - k",
        "local z = ... local k <const> = 0 do local k = 0 return z - k end",
        "local z = ... local k <const> = 0 local k = 0 return z - k",
        "local z = ... local k <const> = 0 return function(k) return z - k end",
        "local z = ... local a <const>, b <const> = 0, 5 return z - a",
        "local z = ... return z - (1.0 // 1 - 1)",
        "local z = ... return z - (2 ^ 0 - 1)",
        "local z = ... return z - (1 // 0)",
        "local z = ... return z - (1 % 0)",
    ] {
        assert!(!has_equal(CompatibilityProfile::Lua54, source), "{source}");
    }
}

fn function_opcodes(source: &str, config: CompilerConfig) -> Vec<OpCode> {
    let chunk = compile_with_config(source, config).unwrap();
    chunk.prototypes[0]
        .code
        .iter()
        .map(|instruction| instruction.opcode().unwrap())
        .collect()
}

#[test]
fn right_numeric_literals_use_the_constant_arithmetic_opcodes() {
    for (operator, opcode) in [
        ("+", OpCode::AddK),
        ("-", OpCode::SubtractK),
        ("*", OpCode::MultiplyK),
        ("/", OpCode::DivideK),
        ("%", OpCode::ModuloK),
    ] {
        for literal in ["3", "2.5"] {
            let source = format!("local function f(x) return x {operator} {literal} end");
            for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
                let opcodes = function_opcodes(&source, CompilerConfig::new(profile));
                assert_eq!(
                    opcodes,
                    vec![opcode, OpCode::Return, OpCode::Halt],
                    "{source} under {profile:?}"
                );
            }
        }
    }
}

#[test]
fn constant_arithmetic_records_the_constant_and_names_the_left_operand() {
    let chunk = compile("local function f(x) return x + 7 end").unwrap();
    let prototype = &chunk.prototypes[0];
    let instruction = prototype.code[0];
    assert_eq!(instruction.opcode(), Ok(OpCode::AddK));
    assert_eq!(instruction.b(), 0);
    assert_eq!(
        prototype.constants[usize::from(instruction.c())],
        Constant::Number(7.0)
    );
    let sites: Vec<_> = prototype
        .debug
        .diagnostic_sites()
        .iter()
        .map(|site| (site.pc(), site.register()))
        .collect();
    assert_eq!(sites, vec![(0, 0)]);
}

#[test]
fn other_operands_keep_the_register_arithmetic_opcodes() {
    let lua51 = CompilerConfig::new(CompatibilityProfile::Lua51);
    // A constant on the left, a string constant, a negated literal and a
    // non-arithmetic operator.
    for source in [
        "local function f(x) return 3 + x end",
        "local function f(x) return x + '3' end",
        "local function f(x) return x + (-3) end",
        "local function f(x) return x ^ 2 end",
    ] {
        let opcodes = function_opcodes(source, lua51);
        assert!(
            opcodes.iter().all(|opcode| !matches!(
                opcode,
                OpCode::AddK
                    | OpCode::SubtractK
                    | OpCode::MultiplyK
                    | OpCode::DivideK
                    | OpCode::ModuloK
            )),
            "{source}: {opcodes:?}"
        );
    }
    let luajit = CompilerConfig::new(CompatibilityProfile::LuaJit21);
    let opcodes = function_opcodes("local function f(x) return x + 3LL end", luajit);
    assert_eq!(
        opcodes,
        vec![
            OpCode::LoadConstant,
            OpCode::Add,
            OpCode::Return,
            OpCode::Halt
        ]
    );
}

#[test]
fn lua54_subtracting_zero_keeps_its_own_sequence() {
    let lua54 = CompilerConfig::new(CompatibilityProfile::Lua54);
    let opcodes = function_opcodes("local function f(x) return x - 0 end", lua54);
    assert!(opcodes.contains(&OpCode::Subtract), "{opcodes:?}");
    assert!(!opcodes.contains(&OpCode::SubtractK), "{opcodes:?}");
}

#[test]
fn a_constant_beyond_index_255_uses_the_register_form() {
    let mut source = String::from("local function f(x) local t = {");
    for index in 0..300 {
        let _ = write!(source, "{index}.5,");
    }
    source.push_str("} return x + 1000.25 end");
    let chunk = compile(&source).unwrap();
    let opcodes: Vec<_> = chunk.prototypes[0]
        .code
        .iter()
        .map(|instruction| instruction.opcode().unwrap())
        .collect();
    assert!(opcodes.contains(&OpCode::Add), "{opcodes:?}");
    assert!(!opcodes.contains(&OpCode::AddK), "{opcodes:?}");
}

fn function_code(source: &str, profile: CompatibilityProfile) -> Vec<crate::bytecode::Instruction> {
    let chunk = compile_with_config(source, CompilerConfig::new(profile)).unwrap();
    chunk.prototypes[0].code.clone()
}

/// The test instructions of `code`, each with the jump word that follows.
fn tests_in(
    code: &[crate::bytecode::Instruction],
) -> Vec<(OpCode, bool, bool, crate::bytecode::Instruction)> {
    code.iter()
        .enumerate()
        .filter(|(_, instruction)| instruction.opcode().unwrap().is_test())
        .map(|(pc, instruction)| {
            let word = code[pc + 1];
            assert_eq!(word.opcode(), Ok(OpCode::Jump), "pc {pc}");
            (
                instruction.opcode().unwrap(),
                instruction.test_expected(),
                instruction.test_constant_left(),
                *instruction,
            )
        })
        .collect()
}

#[test]
fn comparisons_in_conditions_become_tests() {
    let lua54 = CompatibilityProfile::Lua54;
    // (source, opcode, expected, constant_left). The jump is taken when the
    // condition is false, so `expected` is the result that makes it false.
    let cases = [
        ("if a == b then end", OpCode::TestEqual, false, false),
        ("if a ~= b then end", OpCode::TestEqual, true, false),
        ("if a < b then end", OpCode::TestLessThan, false, false),
        ("if a <= b then end", OpCode::TestLessEqual, false, false),
        ("if a > b then end", OpCode::TestLessThan, false, false),
        ("if a >= b then end", OpCode::TestLessEqual, false, false),
        ("if not (a < b) then end", OpCode::TestLessThan, true, false),
        (
            "if not not (a < b) then end",
            OpCode::TestLessThan,
            false,
            false,
        ),
        ("if not (a ~= b) then end", OpCode::TestEqual, false, false),
        ("if ((a == b)) then end", OpCode::TestEqual, false, false),
        ("while a < b do end", OpCode::TestLessThan, false, false),
        ("repeat until a < b", OpCode::TestLessThan, false, false),
        (
            "if x then elseif a <= b then end",
            OpCode::TestLessEqual,
            false,
            false,
        ),
        ("if a == 1 then end", OpCode::TestEqualK, false, false),
        ("if 1 == a then end", OpCode::TestEqualK, false, false),
        ("if a ~= nil then end", OpCode::TestEqualK, true, false),
        ("if a == true then end", OpCode::TestEqualK, false, false),
        ("if a == 'x' then end", OpCode::TestEqualK, false, false),
        ("if a < 1 then end", OpCode::TestLessThanK, false, false),
        ("if 1 < a then end", OpCode::TestLessThanK, false, true),
        ("if a > 1 then end", OpCode::TestLessThanK, false, true),
        ("if 1 > a then end", OpCode::TestLessThanK, false, false),
        ("if a <= 1 then end", OpCode::TestLessEqualK, false, false),
        ("if 1 <= a then end", OpCode::TestLessEqualK, false, true),
        ("if a >= 1 then end", OpCode::TestLessEqualK, false, true),
        ("if 1 >= a then end", OpCode::TestLessEqualK, false, false),
        ("if a < -1 then end", OpCode::TestLessThanK, false, false),
        ("if a < 2.5 then end", OpCode::TestLessThanK, false, false),
        (
            "if not (a >= 1) then end",
            OpCode::TestLessEqualK,
            true,
            true,
        ),
    ];
    for (condition, opcode, expected, constant_left) in cases {
        let source = format!("local function f(a, b, x) {condition} end");
        let code = function_code(&source, lua54);
        let found = tests_in(&code);
        assert_eq!(found.len(), 1, "{condition}: {code:?}");
        assert_eq!(
            (found[0].0, found[0].1, found[0].2),
            (opcode, expected, constant_left),
            "{condition}"
        );
        assert!(
            code.iter().all(|instruction| !matches!(
                instruction.opcode(),
                Ok(OpCode::Equal | OpCode::LessThan | OpCode::LessEqual)
            )),
            "{condition}: {code:?}"
        );
    }
}

#[test]
fn a_greater_than_test_swaps_its_registers_and_keeps_the_evaluation_order() {
    let code = function_code(
        "local function f(a, b) if a > b then end end",
        CompatibilityProfile::Lua54,
    );
    let found = tests_in(&code);
    // `a > b` is `b < a`: the test names register b (1) before a (0).
    assert_eq!((found[0].3.a(), found[0].3.b()), (1, 0));
    // Operands that need evaluation go left to right.
    let code = function_code(
        "local function f(a, b) if g() > h() then end end",
        CompatibilityProfile::Lua54,
    );
    let results: Vec<u8> = code
        .iter()
        .filter(|instruction| matches!(instruction.opcode(), Ok(OpCode::Call | OpCode::CallOne)))
        .map(|instruction| instruction.a())
        .collect();
    assert_eq!(results.len(), 2, "{code:?}");
    let found = tests_in(&code);
    // The first call is the left operand `g()`, and `h() < g()` names the
    // second result first.
    assert_eq!((found[0].3.a(), found[0].3.b()), (results[1], results[0]));
}

#[test]
fn a_test_jump_goes_to_the_false_branch() {
    let code = function_code(
        "local function f(a) local r = 0 if a < 1 then r = 1 else r = 2 end return r end",
        CompatibilityProfile::Lua54,
    );
    let pc = code
        .iter()
        .position(|instruction| instruction.opcode().unwrap().is_test())
        .unwrap();
    let jump = code[pc + 1];
    let target = pc + 2 + usize::try_from(jump.jump_offset()).unwrap();
    // The else arm starts after the then arm's own jump over it.
    assert_eq!(code[target - 1].opcode(), Ok(OpCode::Jump));
    assert_eq!(code[target].opcode(), Ok(OpCode::LoadConstant));
}

#[test]
fn conditions_outside_the_test_forms_keep_the_boolean() {
    let lua54 = CompatibilityProfile::Lua54;
    for source in [
        // Comparisons used as values, and non-comparison conditions.
        "local function f(a, b) local c = a < b return c end",
        "local function f(a, b) if a then end end",
        "local function f(a, b) if not a then end end",
        "local function f(a, b) while f(a) do end end",
        // A condition that spans lines keeps line events as they are.
        "local function f(a, b) if a <\n b then end end",
    ] {
        let code = function_code(source, lua54);
        assert!(tests_in(&code).is_empty(), "{source}: {code:?}");
    }
}

#[test]
fn constants_beyond_index_255_keep_the_register_test() {
    let mut source = String::from("local function f(a) local t = {");
    for index in 0..300 {
        let _ = write!(source, "{index}.5,");
    }
    source.push_str("} if a < 1000.25 then end end");
    let code = function_code(&source, CompatibilityProfile::Lua54);
    let found = tests_in(&code);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, OpCode::TestLessThan);
}

#[test]
fn luajit_64_bit_literals_keep_the_register_test() {
    let code = function_code(
        "local function f(a) if a == 1LL then end end",
        CompatibilityProfile::LuaJit21,
    );
    let found = tests_in(&code);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, OpCode::TestEqual);
}

#[test]
fn tests_name_their_register_operands_for_errors() {
    let chunk = compile("local function f(a, b) if a < b then end end").unwrap();
    let prototype = &chunk.prototypes[0];
    let sites: Vec<_> = prototype
        .debug
        .diagnostic_sites()
        .iter()
        .map(|site| (site.pc(), site.register()))
        .collect();
    assert_eq!(sites, vec![(0, 0), (0, 1)]);
    assert_eq!(prototype.validate(), Ok(()));
}

#[test]
fn a_test_and_its_jump_share_the_condition_line() {
    let chunk =
        compile("local function f(a, b)\n  if a < b then\n    return 1\n  end\nend").unwrap();
    let prototype = &chunk.prototypes[0];
    let lines = prototype.debug.line_info();
    let pc = prototype
        .code
        .iter()
        .position(|instruction| instruction.opcode().unwrap().is_test())
        .unwrap();
    assert_eq!(lines[pc], lines[pc + 1]);
    assert_eq!(lines[pc], 2);
}

/// The branch structure of `code`: each test as `T<expected>`, each
/// `JumpIfFalse` as `F`, and each jump word as `->target` (an absolute pc).
/// Other instructions are skipped.
fn branch_shape(code: &[crate::bytecode::Instruction]) -> String {
    let mut parts = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let instruction = code[pc];
        let opcode = instruction.opcode().unwrap();
        if opcode.is_test() {
            let jump = code[pc + 1];
            let target = i64::try_from(pc + 2).unwrap() + i64::from(jump.jump_offset());
            let flag = if instruction.test_expected() {
                'T'
            } else {
                'F'
            };
            parts.push(format!("test{flag}->{target}"));
            pc += 2;
            continue;
        }
        match opcode {
            OpCode::JumpIfFalse => {
                let offset = i16::from_ne_bytes(instruction.d().to_ne_bytes());
                parts.push(format!(
                    "jif->{}",
                    i64::try_from(pc + 1).unwrap() + i64::from(offset)
                ));
            }
            OpCode::Jump => {
                parts.push(format!(
                    "jump->{}",
                    i64::try_from(pc + 1).unwrap() + i64::from(instruction.jump_offset())
                ));
            }
            _ => {}
        }
        pc += 1;
    }
    parts.join(" ")
}

fn shape_of(statement: &str) -> String {
    let source = format!("local function f(a, b, c, d, x, y) {statement} end");
    branch_shape(&function_code(&source, CompatibilityProfile::Lua54))
}

#[test]
fn and_chains_jump_to_one_false_target() {
    // Bodies are empty, so the false target is the instruction after the
    // last jump word.
    assert_eq!(shape_of("if a < b and c < d then end"), "testF->4 testF->4");
    assert_eq!(
        shape_of("if a < b and c < d and x == y then end"),
        "testF->6 testF->6 testF->6"
    );
    // A non-comparison operand is tested with JumpIfFalse in either place.
    assert_eq!(shape_of("if x and a < b then end"), "jif->3 testF->3");
    assert_eq!(shape_of("if a < b and x then end"), "testF->3 jif->3");
    assert_eq!(shape_of("if x and y then end"), "jif->2 jif->2");
}

#[test]
fn or_chains_jump_to_the_body_when_true() {
    // The first comparison jumps over the second to the body at pc 4.
    assert_eq!(shape_of("if a < b or c < d then end"), "testT->4 testF->4");
    assert_eq!(shape_of("if a < b or x then end"), "testT->3 jif->3");
    assert_eq!(
        shape_of("if a < b or c < d or x == y then end"),
        "testT->6 testT->6 testF->6"
    );
}

#[test]
fn a_negated_and_swaps_the_targets() {
    assert_eq!(
        shape_of("if not (a < b and c < d) then end"),
        "testF->4 testT->4"
    );
    assert_eq!(
        shape_of("if not (a < b or c < d) then end"),
        "testT->4 testT->4"
    );
    assert_eq!(shape_of("if not (x and a < b) then end"), "jif->3 testT->3");
}

#[test]
fn operands_that_cannot_jump_when_true_keep_the_value_form() {
    // `x or ...` and `not (... and x)` need jump-if-true on a plain value.
    for statement in [
        "if x or a < b then end",
        "if a < b and (x or y) then end",
        "if not (a < b and x) then end",
        "if not (x or y) then end",
    ] {
        let shape = shape_of(statement);
        assert!(shape.contains("jif"), "{statement}: {shape}");
    }
}

#[test]
fn nested_trees_chain_without_booleans() {
    assert_eq!(
        shape_of("if (a < b or c < d) and (a == b or x) then end"),
        "testT->4 testF->7 testT->7 jif->7"
    );
    assert_eq!(
        shape_of("if not (a < b and not (c < d or x)) then end"),
        "testF->5 testT->5 jif->5"
    );
}

#[test]
fn while_repeat_and_elseif_use_chains() {
    // The loop exit is after the back jump.
    assert_eq!(
        shape_of("while a < b and c < d do end"),
        "testF->5 testF->5 jump->0"
    );
    assert_eq!(
        shape_of("while a < b or c < d do end"),
        "testT->4 testF->5 jump->0"
    );
    // `repeat` jumps back to the start when the condition is false.
    assert_eq!(
        shape_of("repeat until a < b and c < d"),
        "testF->0 testF->0"
    );
    assert_eq!(shape_of("repeat until a < b or x"), "testT->3 jif->0");
    assert_eq!(
        shape_of("if x then elseif a < b and c < d then end"),
        "jif->2 jump->6 testF->6 testF->6"
    );
}

#[test]
fn conditions_that_span_lines_keep_the_value_form() {
    for statement in [
        "if a < b and\n c < d then end",
        "if a < b\n or c < d then end",
        "while a < b and\n c < d do end",
    ] {
        let shape = shape_of(statement);
        assert!(!shape.contains("test"), "{statement}: {shape}");
        assert!(shape.contains("jif"), "{statement}: {shape}");
    }
}

#[test]
fn a_repeat_with_captured_locals_keeps_the_value_form() {
    // The close of `k` comes between the condition and its branch, which a
    // jump chain has no place for.
    let shape =
        shape_of("repeat local k = a local g = function() return k end until a < b and c < d");
    assert!(!shape.contains("test") && shape.contains("jif"), "{shape}");
    let shape = shape_of("repeat local k = a until a < b and c < d");
    assert!(shape.contains("test") && !shape.contains("jif"), "{shape}");
}
