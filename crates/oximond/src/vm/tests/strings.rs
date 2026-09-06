use super::*;

#[test]
fn format_direct_path_accepts_numeric_conversions_and_defers_tostring() {
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
    install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua54).unwrap();
    assert_eq!(
        vm.try_string_format_direct(&[
            Value::from("%d:%04x:%.1f"),
            Value::Number(12.0),
            Value::Number(15.0),
            Value::Number(2.5),
        ])
        .unwrap(),
        Some(vec![Value::from("12:000f:2.5")]),
    );
    assert_eq!(
        vm.try_string_format_direct(&[Value::from("%s"), Value::Number(12.0)])
            .unwrap(),
        None,
    );
    assert!(
        vm.try_string_format_direct(&[Value::from("%d"), Value::from("no number")])
            .is_err()
    );
}

#[test]
fn format_call_honors_string_tostring_metamethod() {
    fn custom_tostring(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::from("custom")])
    }

    let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
    install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua54).unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    let tostring = vm.runtime_mut().register_native(custom_tostring).unwrap();
    vm.runtime_mut()
        .table_set(
            metatable,
            &Value::from("__tostring"),
            Value::NativeFunction(tostring),
        )
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::from("sample"), Some(metatable))
        .unwrap();
    assert_eq!(
        vm.try_string_format_direct(&[Value::from("%s"), Value::from("sample")])
            .unwrap(),
        None,
    );
    vm.runtime_mut().set_global(
        "format",
        Value::BuiltinFunction(BuiltinFunction::StringFormat),
    );
    let chunk = Chunk::new(
        3,
        vec![
            Constant::String("format".into()),
            Constant::String("%s".into()),
            Constant::String("sample".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::call(0, 2, 1),
            Instruction::return_(0),
        ],
    );
    assert_eq!(vm.execute(&chunk).unwrap(), vec![Value::from("custom")]);
}

#[test]
fn gsub_direct_path_handles_templates_limits_and_callback_fallback() {
    let mut vm = Vm::new();
    assert_eq!(
        vm.try_string_gsub_direct(&[Value::from("a1 b2"), Value::from("%d+"), Value::from("n"),])
            .unwrap(),
        Some(vec![Value::from("an bn"), Value::Number(2.0)]),
    );
    assert_eq!(
        vm.try_string_gsub_direct(&[
            Value::from("a1 b2 c3"),
            Value::from("(%a)(%d)"),
            Value::from("%2%1"),
            Value::Number(2.0),
        ])
        .unwrap(),
        Some(vec![Value::from("1a 2b c3"), Value::Number(2.0)]),
    );
    assert_eq!(
        vm.try_string_gsub_direct(&[Value::from("abc"), Value::from("."), Value::Boolean(true),])
            .unwrap(),
        None,
    );
    assert!(
        vm.try_string_gsub_direct(&[Value::from("abc"), Value::from("."), Value::from("%9"),])
            .is_err()
    );
}

#[test]
fn protected_gsub_reports_direct_template_errors() {
    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    vm.runtime_mut()
        .set_global("gsub", Value::BuiltinFunction(BuiltinFunction::StringGsub));
    let chunk = Chunk::new(
        5,
        vec![
            Constant::String("pcall".into()),
            Constant::String("gsub".into()),
            Constant::String("abc".into()),
            Constant::String(".".into()),
            Constant::String("%9".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(3, 3),
            Instruction::load_constant(4, 4),
            Instruction::call(0, 4, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert!(matches!(
        vm.execute(&chunk).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(_)]
    ));
}

#[test]
fn protected_format_reports_direct_numeric_errors() {
    let mut vm = Vm::with_profile(CompatibilityProfile::Lua54);
    install_base_with_profile(vm.runtime_mut(), CompatibilityProfile::Lua54).unwrap();
    vm.runtime_mut().set_global(
        "format",
        Value::BuiltinFunction(BuiltinFunction::StringFormat),
    );
    let chunk = Chunk::new(
        4,
        vec![
            Constant::String("pcall".into()),
            Constant::String("format".into()),
            Constant::String("%d".into()),
            Constant::String("no number".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(3, 3),
            Instruction::call(0, 3, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert!(matches!(
        vm.execute(&chunk).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(_)]
    ));
}

#[test]
fn table_fast_paths_keep_string_keys_and_errors() {
    let string_key = Chunk::new(
        4,
        vec![Constant::String("name".into()), Constant::Number(9.0)],
        vec![
            Instruction::new_table(0, 0, 1),
            Instruction::load_constant(1, 0),
            Instruction::load_constant(2, 1),
            Instruction::set_table(0, 1, 2),
            Instruction::get_table(3, 0, 1),
            Instruction::return_(3),
        ],
    );
    assert_eq!(
        Vm::new().execute(&string_key).unwrap(),
        vec![Value::Number(9.0)]
    );

    let invalid_key = Chunk::new(
        3,
        vec![Constant::Number(1.0)],
        vec![
            Instruction::new_table(0, 0, 1),
            Instruction::load_nil(1),
            Instruction::load_constant(2, 0),
            Instruction::set_table(0, 1, 2),
            Instruction::return_(0),
        ],
    );
    assert!(matches!(
        Vm::new().execute(&invalid_key),
        Err(VmError::Runtime(RuntimeError::InvalidTableKey("nil")))
    ));

    let non_table = Chunk::new(
        3,
        vec![Constant::Number(1.0), Constant::String("name".into())],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::get_table(2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new().execute(&non_table).unwrap_err().to_string(),
        "attempt to index a number value"
    );
}

#[test]
fn arithmetic_and_numeric_for_convert_numeric_strings_once() {
    let arithmetic = Chunk::new(
        3,
        vec![
            Constant::String(" 2.5 ".into()),
            Constant::String("4".into()),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::binary(OpCode::Add, 2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new().execute(&arithmetic).unwrap(),
        vec![Value::Number(6.5)]
    );

    let loop_with_strings = Chunk::new(
        5,
        vec![
            Constant::String("1".into()),
            Constant::String("3".into()),
            Constant::String("1".into()),
            Constant::Number(0.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::load_constant(2, 2),
            Instruction::load_constant(4, 3),
            Instruction::numeric_for_prep(0, 2),
            Instruction::aj(OpCode::Jump, 2).unwrap(),
            Instruction::binary(OpCode::Add, 4, 4, 3),
            Instruction::numeric_for_loop(0, -2),
            Instruction::return_(4),
        ],
    );
    assert_eq!(
        Vm::new().execute(&loop_with_strings).unwrap(),
        vec![Value::Number(6.0)]
    );
}

#[test]
fn tostring_distinguishes_a_missing_argument_from_nil() {
    let mut vm = Vm::new();
    vm.runtime_mut().set_global(
        "pcall",
        Value::BuiltinFunction(BuiltinFunction::ProtectedCall),
    );
    vm.runtime_mut().set_global(
        "tostring",
        Value::BuiltinFunction(BuiltinFunction::ToString),
    );
    let missing = Chunk::new(
        2,
        vec![
            Constant::String("pcall".into()),
            Constant::String("tostring".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::call(0, 1, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert!(matches!(
        vm.execute(&missing).unwrap().as_slice(),
        [Value::Boolean(false), Value::String(_)]
    ));

    let explicit_nil = Chunk::new(
        3,
        vec![
            Constant::String("pcall".into()),
            Constant::String("tostring".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::get_global(1, 1),
            Instruction::load_nil(2),
            Instruction::call(0, 2, 2),
            Instruction::return_values(0, 2),
        ],
    );
    assert_eq!(
        vm.execute(&explicit_nil).unwrap(),
        vec![Value::Boolean(true), Value::from("nil")]
    );
}

#[test]
fn power_coerces_numeric_strings_and_reports_bad_operands() {
    let power = Chunk::new(
        3,
        vec![Constant::String("9".into()), Constant::String("0.5".into())],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::power(2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(Vm::new().execute(&power).unwrap(), vec![Value::Number(3.0)]);

    let bad_power = Chunk::new(
        3,
        vec![Constant::String("nope".into()), Constant::Number(2.0)],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::power(2, 0, 1),
            Instruction::return_(2),
        ],
    );
    assert_eq!(
        Vm::new().execute(&bad_power).unwrap_err().to_string(),
        "attempt to perform arithmetic on a string value"
    );
}

#[test]
fn strings_resolve_methods_through_their_shared_metatable() {
    fn method(_: &mut Runtime, arguments: &[Value]) -> RuntimeResult<Vec<Value>> {
        if arguments.is_empty() {
            return Err(RuntimeError::message("string method needs a receiver"));
        }
        Ok(arguments.to_vec())
    }

    let chunk = Chunk::new(
        5,
        vec![
            Constant::String("hello".into()),
            Constant::String("sub".into()),
            Constant::Number(1.0),
        ],
        vec![
            Instruction::load_constant(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::get_table(2, 0, 1),
            Instruction::move_(3, 0),
            Instruction::load_constant(4, 2),
            Instruction::call(2, 2, 2),
            Instruction::return_values(2, 2),
        ],
    );
    let mut vm = Vm::new();
    let string = vm.runtime_mut().create_table().unwrap();
    let method = vm.runtime_mut().register_native(method).unwrap();
    vm.runtime_mut()
        .table_set(string, &Value::from("sub"), Value::NativeFunction(method))
        .unwrap();
    let metatable = vm.runtime_mut().create_table().unwrap();
    vm.runtime_mut()
        .table_set(metatable, &Value::from("__index"), Value::Table(string))
        .unwrap();
    vm.runtime_mut()
        .raw_set_metatable(&Value::from(""), Some(metatable))
        .unwrap();
    assert_eq!(
        vm.execute(&chunk).unwrap(),
        vec![Value::from("hello"), Value::Number(1.0)]
    );
}
