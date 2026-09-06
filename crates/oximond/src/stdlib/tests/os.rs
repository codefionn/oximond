use super::*;

#[test]
fn os_getenv_reads_host_values_and_returns_nil_for_absent_names() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(getenv) = runtime.table_get(os, &Value::from("getenv")).unwrap()
    else {
        panic!("os.getenv is native");
    };
    let (name, value) = std::env::vars_os()
        .find_map(|(name, value)| Some((name.into_string().ok()?, value.into_string().ok()?)))
        .expect("host has a Unicode environment variable");
    assert_eq!(
        runtime.call_native(getenv, &[Value::from(name)]).unwrap(),
        vec![Value::from(value)]
    );
    assert_eq!(
        runtime
            .call_native(getenv, &[Value::from("OXIMOND_TEST_MISSING_ENV_498572045")])
            .unwrap(),
        vec![Value::Nil]
    );
    assert!(runtime.call_native(getenv, &[]).is_err());
    assert!(
        runtime
            .call_native(getenv, &[Value::Boolean(true)])
            .is_err()
    );
}

#[test]
fn package_config_exposes_lua_loader_markers() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(package) = runtime.get_global("package") else {
        panic!("package is a table");
    };
    let config = runtime.table_get(package, &Value::from("config")).unwrap();
    #[cfg(unix)]
    assert_eq!(config, Value::from("/\n;\n?\n!\n-\n"));
    #[cfg(windows)]
    assert_eq!(config, Value::from("\\\n;\n?\n!\n-\n"));
}

#[test]
fn os_execute_reports_shell_availability_and_system_style_statuses() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(execute) = runtime.table_get(os, &Value::from("execute")).unwrap()
    else {
        panic!("os.execute is native");
    };
    assert!(matches!(
        runtime.call_native(execute, &[]).unwrap().as_slice(),
        [Value::Number(status)] if *status != 0.0
    ));
    assert_eq!(
        runtime
            .call_native(execute, &[Value::from("exit 0")])
            .unwrap(),
        vec![Value::Number(0.0)]
    );
    assert!(matches!(
        runtime.call_native(execute, &[Value::from("exit 7")]).unwrap().as_slice(),
        [Value::Number(status)] if *status != 0.0
    ));
    assert!(
        runtime
            .call_native(execute, &[Value::Boolean(true)])
            .is_err()
    );
}

#[test]
fn os_setlocale_is_fixed_to_c_without_mutating_the_host() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(setlocale) =
        runtime.table_get(os, &Value::from("setlocale")).unwrap()
    else {
        panic!("os.setlocale is native");
    };
    for arguments in [
        vec![],
        vec![Value::Nil],
        vec![Value::from("C")],
        vec![Value::from("C"), Value::from("all")],
        vec![Value::Nil, Value::from("numeric")],
        vec![
            Value::from("C"),
            Value::from("time"),
            Value::from("ignored"),
        ],
    ] {
        assert_eq!(
            runtime.call_native(setlocale, &arguments).unwrap(),
            vec![Value::from("C")]
        );
    }
    assert_eq!(
        runtime
            .call_native(setlocale, &[Value::from("pt_BR")])
            .unwrap(),
        vec![Value::Nil]
    );
    assert!(
        runtime
            .call_native(setlocale, &[Value::from("C"), Value::from("bogus")])
            .is_err()
    );
}

#[test]
fn os_time_round_trips_local_tables_and_normalizes_calendar_fields() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let native = |name| {
        let Value::NativeFunction(function) = runtime.table_get(os, &Value::from(name)).unwrap()
        else {
            panic!("os.{name} is native");
        };
        function
    };
    let date = native("date");
    let time = native("time");
    let difftime = native("difftime");
    let now = runtime.call_native(time, &[]).unwrap().remove(0);
    let Value::Table(fields) = runtime
        .call_native(date, &[Value::from("*t"), now.clone()])
        .unwrap()
        .remove(0)
    else {
        panic!("os.date *t returns a table");
    };
    assert_eq!(
        runtime.call_native(time, &[Value::Table(fields)]).unwrap(),
        vec![now]
    );

    let first = runtime.create_table().unwrap();
    let second = runtime.create_table().unwrap();
    for (table, minute, second_value) in [(first, 12.0, 17.0), (second, 10.0, 19.0)] {
        for (name, value) in [
            ("year", 2000.0),
            ("month", 10.0),
            ("day", 1.0),
            ("hour", 23.0),
            ("min", minute),
            ("sec", second_value),
        ] {
            runtime
                .table_set(table, &Value::from(name), Value::Number(value))
                .unwrap();
        }
    }
    let first = runtime
        .call_native(time, &[Value::Table(first)])
        .unwrap()
        .remove(0);
    let second = runtime
        .call_native(time, &[Value::Table(second)])
        .unwrap()
        .remove(0);
    assert_eq!(
        runtime.call_native(difftime, &[first, second]).unwrap(),
        vec![Value::Number(118.0)]
    );
}

#[test]
fn os_date_and_difftime_validate_their_inputs() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let native = |name| {
        let Value::NativeFunction(function) = runtime.table_get(os, &Value::from(name)).unwrap()
        else {
            panic!("os.{name} is native");
        };
        function
    };
    let date = native("date");
    let time = native("time");
    let difftime = native("difftime");
    assert!(runtime.call_native(date, &[Value::from("%Q")]).is_err());
    assert!(runtime.call_native(date, &[Value::Boolean(true)]).is_err());
    assert!(matches!(
        runtime
            .call_native(date, &[Value::from("%Y"), Value::Nil])
            .unwrap()
            .as_slice(),
        [Value::String(_)]
    ));
    assert!(matches!(
        runtime.call_native(time, &[Value::Nil]).unwrap().as_slice(),
        [Value::Number(_)]
    ));
    // Lua 5.1 and LuaJIT default the second time to zero.
    assert_eq!(
        runtime
            .call_native(difftime, &[Value::Number(1.0)])
            .unwrap(),
        vec![Value::Number(1.0)]
    );
    assert!(
        runtime
            .call_native(difftime, &[Value::Number(f64::NAN), Value::Number(0.0)])
            .is_err()
    );
}
