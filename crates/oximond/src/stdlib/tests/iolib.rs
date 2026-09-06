use super::*;

#[cfg(unix)]
#[test]
fn io_popen_reads_and_writes_live_process_pipes() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(io) = runtime.get_global("io") else {
        panic!("io is a table");
    };
    let Value::NativeFunction(popen) = runtime.table_get(io, &Value::from("popen")).unwrap() else {
        panic!("io.popen is native");
    };
    let Value::File(reader) = runtime
        .call_native(
            popen,
            &[Value::from("printf 'one\\ntwo\\n'"), Value::from("r")],
        )
        .unwrap()[0]
    else {
        panic!("popen returns a file");
    };
    assert_eq!(runtime.file_read_line(reader).unwrap(), Some("one".into()));
    assert_eq!(runtime.file_read_all(reader).unwrap(), "two\n".into());
    assert!(
        runtime
            .file_close_process(reader)
            .unwrap()
            .unwrap()
            .success()
    );

    let Value::File(writer) = runtime
        .call_native(popen, &[Value::from("cat >/dev/null"), Value::from("w")])
        .unwrap()[0]
    else {
        panic!("popen returns a file");
    };
    runtime.file_write(writer, &[Value::from("hello")]).unwrap();
    assert!(
        runtime
            .file_close_process(writer)
            .unwrap()
            .unwrap()
            .success()
    );
    assert!(
        runtime
            .call_native(popen, &[Value::from("true"), Value::from("rw")])
            .is_err()
    );

    let Value::File(failed) = runtime
        .call_native(popen, &[Value::from("exit 7")])
        .unwrap()[0]
    else {
        panic!("popen returns a file");
    };
    let close = runtime
        .register_native(crate::stdlib::math::file_close)
        .unwrap();
    assert_eq!(
        runtime.call_native(close, &[Value::File(failed)]).unwrap(),
        vec![Value::Boolean(true)]
    );

    runtime.set_lines_formats(crate::runtime::LinesFormats::Lua54);
    let Value::File(failed) = runtime
        .call_native(popen, &[Value::from("exit 7")])
        .unwrap()[0]
    else {
        panic!("popen returns a file");
    };
    assert_eq!(
        runtime.call_native(close, &[Value::File(failed)]).unwrap(),
        vec![Value::Nil, Value::from("exit"), runtime.integer_value(7)]
    );
}

#[test]
fn io_tmpfile_installs_an_opaque_runtime_file() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(io) = runtime.get_global("io") else {
        panic!("io is a table");
    };
    let Value::NativeFunction(tmpfile) = runtime.table_get(io, &Value::from("tmpfile")).unwrap()
    else {
        panic!("io.tmpfile is native");
    };
    let file = runtime.call_native(tmpfile, &[]).unwrap();
    assert!(matches!(file.as_slice(), [Value::File(_)]));
}

#[test]
fn io_named_defaults_write_read_close_and_preserve_bytes() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(io) = runtime.get_global("io") else {
        panic!("io is a table");
    };
    let native = |name| {
        let Value::NativeFunction(function) = runtime.table_get(io, &Value::from(name)).unwrap()
        else {
            panic!("io.{name} is native");
        };
        function
    };
    let output = native("output");
    let input = native("input");
    let write = native("write");
    let read = native("read");
    let close = native("close");
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(tmpname) = runtime.table_get(os, &Value::from("tmpname")).unwrap()
    else {
        panic!("os.tmpname is native");
    };
    let path = runtime.call_native(tmpname, &[]).unwrap().remove(0);

    let selected = runtime
        .call_native(output, std::slice::from_ref(&path))
        .unwrap();
    let Value::File(output_file) = selected[0] else {
        panic!("io.output returns a file");
    };
    assert_eq!(
        runtime
            .call_native(write, &[Value::bytes(vec![0xff, b'x']), Value::Number(2.5)])
            .unwrap(),
        vec![Value::File(output_file)]
    );
    assert_eq!(
        runtime.call_native(close, &[]).unwrap(),
        vec![Value::Boolean(true)]
    );

    let selected = runtime
        .call_native(input, std::slice::from_ref(&path))
        .unwrap();
    let Value::File(input_file) = selected[0] else {
        panic!("io.input returns a file");
    };
    assert_eq!(
        runtime.call_native(input, &[]).unwrap(),
        vec![Value::File(input_file)]
    );
    assert_eq!(
        runtime.call_native(read, &[Value::from("*a")]).unwrap(),
        vec![Value::bytes(vec![0xff, b'x', b'2', b'.', b'5'])]
    );
    assert_eq!(
        runtime
            .call_native(close, &[Value::File(input_file)])
            .unwrap(),
        vec![Value::Boolean(true)]
    );
    assert!(matches!(
        runtime
            .call_native(output, &[Value::bytes(vec![0])])
            .unwrap()
            .as_slice(),
        [Value::Nil, Value::String(_)]
    ));
    let Value::String(path) = path else {
        panic!("os.tmpname returns a string");
    };
    std::fs::remove_file(path.to_lossy().into_owned()).unwrap();
}

#[test]
fn io_write_clock_and_gcinfo_use_runtime_owned_state() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(io) = runtime.get_global("io") else {
        panic!("io is a table");
    };
    let Value::NativeFunction(write) = runtime.table_get(io, &Value::from("write")).unwrap() else {
        panic!("io.write is native");
    };
    let Value::File(stdout) = runtime.table_get(io, &Value::from("stdout")).unwrap() else {
        panic!("io.stdout is a file");
    };
    assert_eq!(
        runtime
            .call_native(write, &[Value::bytes(vec![0xff]), Value::Number(2.5)])
            .unwrap(),
        vec![Value::File(stdout)]
    );
    assert_eq!(runtime.take_output(), vec![0xff, b'2', b'.', b'5']);
    assert!(runtime.call_native(write, &[Value::Boolean(true)]).is_err());

    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(clock) = runtime.table_get(os, &Value::from("clock")).unwrap() else {
        panic!("os.clock is native");
    };
    let before = runtime.call_native(clock, &[]).unwrap();
    let after = runtime.call_native(clock, &[]).unwrap();
    assert!(matches!(
        (before.as_slice(), after.as_slice()),
        ([Value::Number(before)], [Value::Number(after)]) if after >= before
    ));

    let Value::NativeFunction(gcinfo) = runtime.get_global("gcinfo") else {
        panic!("gcinfo is native");
    };
    let values = runtime.call_native(gcinfo, &[]).unwrap();
    assert!(matches!(
        values.as_slice(),
        [Value::Number(kilobytes), Value::Number(remainder)]
            if *kilobytes >= 0.0 && *remainder >= 0.0 && *remainder < 1024.0
    ));
}

#[test]
fn os_tmpname_reserves_distinct_paths_for_caller_owned_cleanup() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(tmpname) = runtime.table_get(os, &Value::from("tmpname")).unwrap()
    else {
        panic!("os.tmpname is native");
    };
    let Value::String(first) = runtime.call_native(tmpname, &[]).unwrap().remove(0) else {
        panic!("os.tmpname returns a string");
    };
    let Value::String(second) = runtime.call_native(tmpname, &[]).unwrap().remove(0) else {
        panic!("os.tmpname returns a string");
    };
    assert_ne!(first, second);
    let first_path = std::path::PathBuf::from(first.to_lossy().into_owned());
    let second_path = std::path::PathBuf::from(second.to_lossy().into_owned());
    assert!(first_path.is_file());
    assert!(second_path.is_file());
    std::fs::remove_file(first_path).unwrap();
    std::fs::remove_file(second_path).unwrap();
}

#[test]
fn os_remove_and_rename_return_lua_results_for_host_files() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::Table(os) = runtime.get_global("os") else {
        panic!("os is a table");
    };
    let Value::NativeFunction(tmpname) = runtime.table_get(os, &Value::from("tmpname")).unwrap()
    else {
        panic!("os.tmpname is native");
    };
    let Value::NativeFunction(rename) = runtime.table_get(os, &Value::from("rename")).unwrap()
    else {
        panic!("os.rename is native");
    };
    let Value::NativeFunction(remove) = runtime.table_get(os, &Value::from("remove")).unwrap()
    else {
        panic!("os.remove is native");
    };
    let from = runtime.call_native(tmpname, &[]).unwrap().remove(0);
    let to = runtime.call_native(tmpname, &[]).unwrap().remove(0);
    let Value::String(to_name) = &to else {
        panic!("os.tmpname returns a string");
    };
    std::fs::remove_file(to_name.to_lossy().into_owned()).unwrap();

    assert_eq!(
        runtime
            .call_native(rename, &[from.clone(), to.clone()])
            .unwrap(),
        vec![Value::Boolean(true)]
    );
    assert_eq!(
        runtime
            .call_native(remove, std::slice::from_ref(&to))
            .unwrap(),
        vec![Value::Boolean(true)]
    );
    assert!(matches!(
        runtime.call_native(remove, &[to]).unwrap().as_slice(),
        [Value::Nil, Value::String(_)]
    ));
}

#[test]
fn print_writes_lua_values_as_tab_separated_bytes() {
    let mut runtime = Runtime::new();
    install_base(&mut runtime).unwrap();
    let Value::NativeFunction(print) = runtime.get_global("print") else {
        panic!("print is native");
    };
    let table = runtime.create_table().unwrap();
    runtime
        .call_native(
            print,
            &[
                Value::Nil,
                Value::from(false),
                Value::from(1.25),
                Value::from(&b"a\xffb"[..]),
                Value::Table(table),
                Value::NativeFunction(print),
            ],
        )
        .unwrap();
    let mut expected = b"nil\tfalse\t1.25\ta".to_vec();
    expected.push(0xff);
    expected.extend_from_slice(
        format!(
            "b\ttable: {}\tfunction: native {}\n",
            table.index(),
            print.index()
        )
        .as_bytes(),
    );
    assert_eq!(runtime.take_output(), expected);

    runtime.call_native(print, &[]).unwrap();
    assert_eq!(runtime.take_output(), b"\n");
}
