use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use oximond::bytecode::{Chunk, Constant, Instruction, binary};

const USAGE_EXIT: i32 = 64;
const IO_EXIT: i32 = 66;
const COMPILE_EXIT: i32 = 65;
const RUNTIME_EXIT: i32 = 70;

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(arguments)
        .output()
        .expect("run oximond")
}

fn lua55_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oximond"));
    command
        .env_remove("LUA_INIT")
        .env_remove("LUA_INIT_5_5")
        .env_remove("LUA_PATH")
        .env_remove("LUA_PATH_5_5")
        .args(["--compat", "lua55"]);
    command
}

fn temporary_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time")
        .as_nanos();
    env::temp_dir().join(format!(
        "oximond-cli-{name}-{}-{nonce}.lua",
        std::process::id()
    ))
}

#[test]
fn trace_profiles_record_after_guest_error_and_load_on_later_run() {
    let profile = temporary_path("trace-profile");
    let path = profile.to_str().expect("UTF-8 test path");
    let source = "local total = 0; for i = 1, 300 do total = total + i end; error('done')";
    let recorded = run(&[
        "--execution",
        "tracing",
        "--trace-profile-record",
        path,
        "-e",
        source,
    ]);
    assert_eq!(recorded.status.code(), Some(RUNTIME_EXIT));
    let bytes = fs::read(&profile).expect("guest error should still leave a trace profile");
    assert_ne!(bytes, []);

    let replayed = run(&[
        "--execution",
        "tracing",
        "--trace-profile-load",
        path,
        "-e",
        source,
    ]);
    assert_eq!(replayed.status.code(), Some(RUNTIME_EXIT));
    assert_eq!(replayed.stdout, recorded.stdout);
    fs::remove_file(profile).expect("remove trace profile");

    let exit_profile = temporary_path("trace-profile-exit");
    let exit_path = exit_profile.to_str().expect("UTF-8 test path");
    let exited = run(&[
        "--execution",
        "tracing",
        "--trace-profile-record",
        exit_path,
        "-e",
        "os.exit(17)",
    ]);
    assert_eq!(exited.status.code(), Some(17));
    assert_ne!(
        fs::read(&exit_profile).expect("exit should leave a trace profile"),
        []
    );
    fs::remove_file(exit_profile).expect("remove exit trace profile");
}

#[test]
fn loaded_trace_profiles_enter_native_code_without_recording_again() {
    let source = "local gain = 7; local function add(x) return x + gain end; local total = 0; for i = 1, 300 do total = total + add(i) end; print(total)";
    for compatibility in ["lua51", "lua54", "luajit21"] {
        let profile = temporary_path(compatibility);
        let path = profile.to_str().expect("UTF-8 test path");
        let recorded = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .env("OXIMOND_JIT_SYNC", "1")
            .args([
                "--compat",
                compatibility,
                "--execution",
                "tracing",
                "--trace-profile-record",
                path,
                "-e",
                source,
            ])
            .output()
            .expect("record trace profile");
        assert!(recorded.status.success(), "{compatibility}: {recorded:?}");

        let replayed = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .env("OXIMOND_JIT_SYNC", "1")
            .env("OXIMOND_JIT_STATS", "1")
            .args([
                "--compat",
                compatibility,
                "--execution",
                "tracing",
                "--trace-profile-load",
                path,
                "-e",
                source,
            ])
            .output()
            .expect("load trace profile");
        assert!(replayed.status.success(), "{compatibility}: {replayed:?}");
        assert_eq!(replayed.stdout, recorded.stdout, "{compatibility}");
        let stderr = String::from_utf8_lossy(&replayed.stderr);
        assert!(
            stderr.contains("recording_starts: 0"),
            "{compatibility}: {stderr}"
        );
        assert!(
            stderr.contains("native_entries: "),
            "{compatibility}: {stderr}"
        );
        assert!(
            !stderr.contains("native_entries: 0"),
            "{compatibility}: {stderr}"
        );
        fs::remove_file(profile).expect("remove trace profile");
    }
}

#[test]
fn evaluates_an_inline_chunk_without_printing_results() {
    let output = run(&["-e", "local answer = 40 + 2"]);

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, []);
    assert_eq!(output.stderr, []);
}

#[test]
fn streams_flushed_output_before_a_blocking_command_finishes() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["-e", "print('ready'); io.stdout:flush(); io.read('*l')"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start oximond");
    let mut stdout = child.stdout.take().expect("capture standard output");
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut ready = [0; 6];
        let result = std::io::Read::read_exact(&mut stdout, &mut ready).map(|()| ready);
        sender.send(result).expect("report standard output");
    });

    let ready = match receiver.recv_timeout(Duration::from_secs(2)) {
        Ok(ready) => ready.expect("read flushed output"),
        Err(error) => {
            child.kill().expect("stop oximond after output timeout");
            child.wait().expect("reap oximond after output timeout");
            reader.join().expect("join output reader");
            panic!("flushed output did not arrive while oximond waited for input: {error}");
        }
    };
    assert_eq!(&ready, b"ready\n");

    writeln!(child.stdin.take().expect("open standard input"), "continue")
        .expect("release oximond");
    assert!(child.wait().expect("wait for oximond").success());
    reader.join().expect("join output reader");
}

#[test]
fn streams_flushed_standard_error_before_a_blocking_command_finishes() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args([
            "-e",
            "io.stderr:write('ready'); io.stderr:flush(); io.read('*l')",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start oximond");
    let mut stderr = child.stderr.take().expect("capture standard error");
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut ready = [0; 5];
        let result = std::io::Read::read_exact(&mut stderr, &mut ready).map(|()| ready);
        sender.send(result).expect("report standard error");
    });

    let ready = match receiver.recv_timeout(Duration::from_secs(2)) {
        Ok(ready) => ready.expect("read flushed error output"),
        Err(error) => {
            child
                .kill()
                .expect("stop oximond after error output timeout");
            child
                .wait()
                .expect("reap oximond after error output timeout");
            reader.join().expect("join error reader");
            panic!("flushed error output did not arrive while oximond waited for input: {error}");
        }
    };
    assert_eq!(&ready, b"ready");

    writeln!(child.stdin.take().expect("open standard input"), "continue")
        .expect("release oximond");
    assert!(child.wait().expect("wait for oximond").success());
    reader.join().expect("join error reader");
}

#[test]
fn interactive_execution_releases_standard_input_for_io_read() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg("-i")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start interactive oximond");
    let mut stdin = child.stdin.take().expect("open standard input");
    let mut stdout = child.stdout.take().expect("capture standard output");
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for length in [2, 6, 7, 2] {
            let mut bytes = vec![0; length];
            let result = std::io::Read::read_exact(&mut stdout, &mut bytes).map(|()| bytes);
            if sender.send(result).is_err() {
                return;
            }
        }
        // Keep the pipe open for the final newline emitted when stdin closes.
        std::io::copy(&mut stdout, &mut std::io::sink()).expect("drain interactive output");
    });
    let receive =
        |child: &mut std::process::Child| match receiver.recv_timeout(Duration::from_secs(2)) {
            Ok(result) => result.expect("read interactive output"),
            Err(error) => {
                child
                    .kill()
                    .expect("stop interactive oximond after output timeout");
                child
                    .wait()
                    .expect("reap interactive oximond after output timeout");
                panic!("interactive output did not arrive: {error}");
            }
        };

    assert_eq!(receive(&mut child), b"> ");
    writeln!(
        stdin,
        "print('ready'); io.stdout:flush(); print(io.read('*l'))"
    )
    .expect("send interactive command");
    assert_eq!(receive(&mut child), b"ready\n");
    writeln!(stdin, "answer").expect("answer Lua input request");
    assert_eq!(receive(&mut child), b"answer\n");

    drop(stdin);
    assert_eq!(receive(&mut child), b"> ");
    assert!(
        child
            .wait()
            .expect("wait for interactive oximond")
            .success()
    );
    reader.join().expect("join interactive output reader");
}

#[test]
fn reports_inline_runtime_failures() {
    let output = run(&["-e", "assert(false)"]);

    assert_eq!(output.status.code(), Some(RUNTIME_EXIT));
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("runtime error: (command line):1: assertion failed!")
    );
}

#[test]
fn reports_compile_failures_separately() {
    let output = run(&["-e", "local ="]);

    assert_eq!(output.status.code(), Some(COMPILE_EXIT));
    assert!(String::from_utf8_lossy(&output.stderr).contains("compile error:"));
}

#[test]
fn loads_valid_source_and_exposes_iterator_functions() {
    let output = run(&[
        "-e",
        "assert(type(next) == \"function\")\n\
         assert(type(pairs) == \"function\")\n\
         assert(type(ipairs) == \"function\")\n\
         local pair_iterator = pairs({})\n\
         local array_iterator = ipairs({})\n\
         assert(type(pair_iterator) == \"function\")\n\
         assert(type(array_iterator) == \"function\")\n\
         local chunk = load(\"return 40 + 2\")\n\
         assert(type(chunk) == \"function\")\n\
         assert(chunk() == 42)\n\
         local string_chunk = loadstring(\"return 21 + 21\")\n\
         assert(type(string_chunk) == \"function\")\n\
         assert(string_chunk() == 42)",
    ]);

    assert!(output.status.success(), "{output:?}");
}

#[test]
fn dynamically_loaded_chunks_remain_main_chunks_in_debug_info() {
    let output = run(&[
        "--compat",
        "lua51",
        "-e",
        "local source = [[return debug.getinfo(1, 'S').what]]\n\
         local loaded = assert(loadstring(source))\n\
         local dumped = assert(loadstring(string.dump(loaded)))\n\
         assert(dumped() == 'main')",
    ]);

    assert!(output.status.success(), "{output:?}");
}

#[test]
fn automatic_gc_rechecks_heap_after_a_collected_frame_returns() {
    let output = run(&[
        "--compat",
        "lua51",
        "-e",
        "local function inflate()\n\
           local chain = {}\n\
           for i = 1, 20000 do chain = {chain} end\n\
           collectgarbage('stop')\n\
           collectgarbage()\n\
         end\n\
         inflate()\n\
         local weak = {[1] = {}}\n\
         setmetatable(weak, {__mode = 'kv'})\n\
         local allocations = 0\n\
         while weak[1] do\n\
           local garbage = allocations..allocations..allocations..allocations\n\
           allocations = allocations + 1\n\
           assert(allocations < 100000)\n\
         end\n\
         assert(weak[1] == nil)",
    ]);

    assert!(output.status.success(), "{output:?}");
}

#[test]
fn tail_call_placeholders_cannot_redirect_setfenv_to_a_parent_frame() {
    let output = run(&[
        "--compat",
        "lua51",
        "-e",
        "local parent = getfenv(0)\n\
         local replacement = setmetatable({}, {__index = parent})\n\
         local function caller()\n\
           local function callee()\n\
             return setfenv(2, replacement)\n\
           end\n\
           return callee()\n\
         end\n\
         assert(not pcall(caller))\n\
         assert(getfenv(0) == parent)\n\
         assert(getfenv(1) == parent)\n\
         assert(package ~= nil)\n\
         assert(require ~= nil)",
    ]);

    assert!(output.status.success(), "{output:?}");
}

#[test]
fn returns_nil_and_a_message_for_invalid_dynamic_source() {
    let output = run(&[
        "-e",
        "local chunk, message = load(\"local =\")\n\
         assert(chunk == nil)\n\
         assert(type(message) == \"string\")\n\
         local string_chunk, string_message = loadstring(\"local =\")\n\
         assert(string_chunk == nil)\n\
         assert(type(string_message) == \"string\")",
    ]);

    assert!(output.status.success(), "{output:?}");
}

#[test]
fn reports_usage_and_file_errors_separately() {
    let usage = run(&["--no-such-option"]);
    assert_eq!(usage.status.code(), Some(USAGE_EXIT));
    assert!(String::from_utf8_lossy(&usage.stderr).contains("Usage: oximond"));

    let missing = run(&["does-not-exist.lua"]);
    assert_eq!(missing.status.code(), Some(IO_EXIT));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("cannot read"));
}

#[test]
fn runs_the_pinned_assignment_file() {
    let path = temporary_path("assignment");
    fs::write(&path, include_str!("fixtures/assignment.lua")).expect("write fixture");

    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg(&path)
        .arg("ignored-program-argument")
        .output()
        .expect("run oximond");
    fs::remove_file(&path).expect("remove fixture");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, []);
    assert_eq!(output.stderr, []);
}

#[test]
fn runs_non_utf8_short_and_long_string_source_files() {
    for (name, source) in [
        ("latin1-short", b"value = '\xff'".as_slice()),
        ("latin1-long", b"value = [[\xff]]".as_slice()),
    ] {
        let path = temporary_path(name);
        fs::write(&path, source).expect("write byte source");
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .arg(&path)
            .output()
            .expect("run oximond");
        fs::remove_file(&path).expect("remove fixture");

        assert!(output.status.success(), "{name}: {output:?}");
    }
}

#[test]
fn loads_a_shebang_prefixed_binary_chunk_without_a_synthetic_newline() {
    let path = temporary_path("binary-shebang");
    let chunk = Chunk::new(
        2,
        vec![
            Constant::String("print".into()),
            Constant::String("binary".into()),
        ],
        vec![
            Instruction::get_global(0, 0),
            Instruction::load_constant(1, 1),
            Instruction::call(0, 1, 0),
            Instruction::halt(),
        ],
    );
    let mut source = b"# shell comment\r\n".to_vec();
    source.extend(binary::dump(&chunk).unwrap());
    fs::write(&path, source).expect("write binary fixture");
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg(&path)
        .output()
        .expect("run oximond");
    fs::remove_file(&path).expect("remove fixture");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"binary\n");
}

#[test]
fn lua55_accepts_a_utf8_bom_before_a_shebang() {
    let path = temporary_path("lua55-bom-shebang");
    fs::write(&path, b"\xef\xbb\xbf#! /usr/bin/env lua\nprint('bom')\n")
        .expect("write BOM and shebang fixture");
    let output = lua55_command().arg(&path).output().expect("run oximond");
    fs::remove_file(&path).expect("remove fixture");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"bom\n");
}

#[test]
fn lua55_prefers_versioned_init_and_path_environment_variables() {
    let generic_module = temporary_path("lua55-generic-path");
    let versioned_module = temporary_path("lua55-versioned-path");
    fs::write(&generic_module, "path_value = 'generic'").expect("write generic module");
    fs::write(&versioned_module, "path_value = 'versioned'").expect("write versioned module");

    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .env("LUA_INIT", "print('generic init')")
        .env("LUA_INIT_5_5", "print('versioned init')")
        .env("LUA_PATH", &generic_module)
        .env("LUA_PATH_5_5", &versioned_module)
        .args(["--compat", "lua55", "-l", "module_name", "-e"])
        .arg("print(path_value)")
        .output()
        .expect("run oximond");
    fs::remove_file(&generic_module).expect("remove generic module");
    fs::remove_file(&versioned_module).expect("remove versioned module");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"versioned init\nversioned\n");
}

#[test]
fn lua55_e_option_ignores_startup_environment_variables() {
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .env("LUA_INIT", "error('generic init must be ignored')")
        .env("LUA_INIT_5_5", "error('versioned init must be ignored')")
        .args(["--compat", "lua55", "-E", "-e"])
        .arg("print('clean startup')")
        .output()
        .expect("run oximond");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"clean startup\n");
}

#[test]
fn lua55_w_option_enables_warnings() {
    let disabled = lua55_command()
        .args(["-e", "warn('quiet')"])
        .output()
        .expect("run oximond");
    assert!(disabled.status.success(), "{disabled:?}");
    assert_eq!(disabled.stderr, []);

    let enabled = lua55_command()
        .args(["-W", "-e", "warn('loud')"])
        .output()
        .expect("run oximond");
    assert!(enabled.status.success(), "{enabled:?}");
    assert_eq!(enabled.stderr, b"Lua warning: loud\n");
}

#[test]
fn executes_options_in_order_and_exposes_script_arguments() {
    let path = temporary_path("arguments");
    fs::write(
        &path,
        "print(arg[0]); print(arg[1]); print(arg[2]); print(arg[3]); print(arg[-1]); print(arg[-2]); print(arg[-3]); print(select('#', ...))",
    )
    .expect("write fixture");
    let source = "print('from -e')";
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg("-e")
        .arg(source)
        .arg("--")
        .arg(&path)
        .arg("first")
        .arg("second")
        .arg("-after-script")
        .output()
        .expect("run oximond");
    fs::remove_file(&path).expect("remove fixture");

    assert!(output.status.success(), "{output:?}");
    let expected = format!(
        "from -e\n{}\nfirst\nsecond\n-after-script\n--\n{source}\n-e\n3\n",
        path.display()
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
}

#[test]
fn runs_a_stdin_script_and_preserves_its_arguments() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg("-")
        .arg("argument")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start oximond");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(b"print(arg[0]); print(arg[1]); print(...)\n")
        .expect("write source");
    let output = child.wait_with_output().expect("wait for oximond");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"-\nargument\nargument\n");
}

#[test]
fn supports_attached_options_and_non_tty_interactive_input() {
    let output = run(&["-v", "-eprint('attached')"]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"Lua 5.1\nattached\n");

    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg("-i")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start interactive oximond");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(b"= 2 + 3\n")
        .expect("write interactive source");
    let interactive = child.wait_with_output().expect("wait for oximond");
    assert!(interactive.status.success(), "{interactive:?}");
    assert_eq!(interactive.stdout, b"> 5\n> \n");
}

/// Runs an interactive session over piped standard input.
fn interact(arguments: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start oximond");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(input.as_bytes())
        .expect("write session input");
    child.wait_with_output().expect("wait for oximond")
}

#[test]
fn interactive_session_survives_errors() {
    let session = interact(
        &["--compat=lua51", "-i"],
        "x = 1\nerror('boom')\nx x\nx = )\nprint(x)\nerror({})\n",
    );
    assert!(session.status.success(), "{session:?}");
    assert_eq!(session.stdout, b"> > > > > 1\n> > \n");
    let errors = String::from_utf8_lossy(&session.stderr);
    assert!(
        errors.starts_with("stdin:1: boom\nstack traceback:\n"),
        "{errors}"
    );
    assert!(errors.contains("\tstdin:1: in main chunk\n"), "{errors}");
    assert!(!errors.contains("xpcall"), "{errors}");
    assert!(errors.contains("near 'x'\n"), "{errors}");
    assert!(errors.contains("near ')'\n"), "{errors}");
    assert!(
        errors.contains("(error object is a table value)\n"),
        "{errors}"
    );
}

#[test]
fn interactive_session_prints_returned_values() {
    // Every profile prints what a chunk returns and takes `=expression`.
    let session = interact(
        &["--compat=lua51", "-i"],
        "return 1, nil\n= 'a', 2\nfor i = 1, 2 do\nprint(i)\nend\n",
    );
    assert!(session.status.success(), "{session:?}");
    assert_eq!(session.stdout, b"> 1\tnil\n> a\t2\n> >> >> 1\n2\n> \n");

    // An expression alone is a syntax error before Lua 5.3.
    let session = interact(&["--compat=lua51", "-i"], "1 + 1\n");
    assert_eq!(session.stdout, b"> > \n");
    assert!(
        String::from_utf8_lossy(&session.stderr).contains("near '1"),
        "{session:?}"
    );

    let session = interact(
        &["--compat=lua54", "-i"],
        "1 + 1\nx = 5\nx, x * 2 -- twice\nstring.rep('a', 3)\nprint = nil\nx\n",
    );
    assert!(session.status.success(), "{session:?}");
    assert_eq!(session.stdout, b"> 2\n> > 5\t10\n> aaa\n> > > \n");
    assert_eq!(
        String::from_utf8_lossy(&session.stderr),
        "error calling 'print' (attempt to call a nil value)\n"
    );
}

#[test]
fn interactive_session_ends_on_exit_and_reports_an_unfinished_chunk() {
    let session = interact(&["-i"], "os.exit(3)\nprint('unreached')\n");
    assert_eq!(session.status.code(), Some(3));
    assert_eq!(session.stdout, b"> ");

    let session = interact(&["-i"], "for i = 1, 2 do");
    assert!(session.status.success(), "{session:?}");
    assert_eq!(session.stdout, b"> >> \n");
    assert!(
        String::from_utf8_lossy(&session.stderr).contains("near '<eof>'"),
        "{session:?}"
    );
}

#[test]
fn bare_command_line_runs_standard_input_when_it_is_not_a_terminal() {
    let script = interact(&[], "print(arg[0], ...)\n");
    assert!(script.status.success(), "{script:?}");
    assert_eq!(script.stdout, b"-\n");

    // `-l` alone still leaves the script to standard input, as in `lua`.
    let script = interact(
        &["--compat=lua54", "-lstring"],
        "print(#string.rep('a', 4))\n",
    );
    assert!(script.status.success(), "{script:?}");
    assert_eq!(script.stdout, b"4\n");

    // `-e` and `-v` run alone.
    let inline = interact(&["-e", "print('inline')"], "print('stdin')\n");
    assert_eq!(inline.stdout, b"inline\n");
}

#[test]
fn preserves_option_tokens_at_negative_arg_indexes() {
    let path = temporary_path("negative-arguments");
    fs::write(
        &path,
        "print(arg[-1]); print(arg[-2]); print(arg[-3] ~= nil); print(arg[0])",
    )
    .expect("write fixture");
    let attached_empty_chunk = ["-e", " "].concat();
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg(attached_empty_chunk)
        .arg("--")
        .arg(&path)
        .output()
        .expect("run oximond");
    fs::remove_file(&path).expect("remove fixture");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("--\n-e \ntrue\n{}\n", path.display())
    );
}

#[test]
fn library_actions_run_before_the_script_installs_arg() {
    let module = temporary_path("arg-module");
    let script = temporary_path("arg-script");
    fs::write(&module, "assert(arg == nil); return true").expect("write module");
    fs::write(&script, "assert(type(arg) == 'table'); print(arg[1], ...)").expect("write script");
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .env("LUA_PATH", &module)
        .arg("-l")
        .arg("module_name")
        .arg(&script)
        .arg("argument")
        .output()
        .expect("run oximond");
    fs::remove_file(&module).expect("remove module");
    fs::remove_file(&script).expect("remove script");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"argument\targument\n");
}

fn run_with_stdin(arguments: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start oximond");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(input)
        .expect("write standard input");
    child.wait_with_output().expect("wait for oximond")
}

/// Number reads peek at their look-ahead byte, so they work on a pipe and
/// leave it, and the text of a failed read, where stock Lua's `ungetc` does.
const STDIN_NUMBERS: &[u8] = b"12 x\n  abc\n0x1Fg\n-.5e1-\n";

#[test]
fn lua51_reads_numbers_from_piped_standard_input() {
    let output = run_with_stdin(
        &[
            "--compat=lua51",
            "-e",
            r#"print(io.read("*n"), ("%q"):format(io.read("*l")))
            print(io.read("*n"))
            print(io.read(1), io.read(0))
            print(io.read("*l"))
            print(io.read("*n", "*l"))
            print(io.read("*n"), io.read(1), io.read("*l"))
            print(io.read(0), io.read("*n"), io.read(0))"#,
        ],
        STDIN_NUMBERS,
    );

    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "12\t\" x\"\nnil\na\t\nbc\n31\tg\n-5\t-\t\nnil\tnil\tnil\n"
    );
}

#[test]
fn lua54_and_lua55_read_numerals_from_piped_standard_input() {
    for profile in ["--compat=lua54", "--compat=lua55"] {
        let output = run_with_stdin(
            &[
                profile,
                "-e",
                r#"local n = io.read("n")
                print(n, math.type(n), ("%q"):format(io.read("l")))
                print(io.read("n"))
                print(io.read(1), io.read(0))
                print(io.read("l"))
                print(io.read("n", "l"))
                print(io.read("n"), io.read(1), io.read("l"))
                print(io.read(0), io.read("n"), io.read(0))"#,
            ],
            STDIN_NUMBERS,
        );

        assert!(output.status.success(), "{profile}: {output:?}");
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "12\tinteger\t\" x\"\nnil\na\t\nbc\n31\tg\n-5.0\t-\t\nnil\tnil\tnil\n",
            "{profile}"
        );
    }
}

#[test]
fn lua52_and_lua53_error_object_tostring_sees_failed_main() {
    for profile in ["lua52", "lua53"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args([
                "--compat",
                profile,
                "-e",
                r"m = {x = 0}
setmetatable(m, {__tostring = function(x)
  collectgarbage()
  return tostring(debug.getinfo(4).currentline + x.x)
end})
error(m)",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(70));
        assert_eq!(
            String::from_utf8_lossy(&output.stderr),
            "oximond: runtime error: 6\n"
        );
    }
}

// Keep the input writer open while checking for a parse failure. A loader that
// reads to EOF, or fills its entire read buffer, cannot pass this check.
fn rejects_before_input_closes(arguments: &[&str], source: &[u8]) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(arguments)
        .env_remove("LUA_INIT")
        .env_remove("LUA_INIT_5_5")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start streaming parser");
    let mut input = child.stdin.take().expect("open source input");
    input.write_all(source).expect("send invalid source");
    input.flush().expect("flush invalid source");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().expect("check streaming parser").is_some() {
            break;
        }
        if std::time::Instant::now() >= deadline {
            child.kill().expect("stop parser waiting for EOF");
            child.wait().expect("reap parser waiting for EOF");
            panic!("parser waited for more bytes after a complete syntax error");
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().expect("collect parser diagnostic");
    drop(input);
    assert_eq!(output.status.code(), Some(COMPILE_EXIT), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("expected"),
        "{output:?}"
    );
}

#[test]
fn stdin_syntax_errors_arrive_before_eof() {
    for prefix in [b"".as_slice(), b"\xef\xbb\xbf# shell comment\r\n"] {
        let mut source = prefix.to_vec();
        source.extend_from_slice(b"local = 1\n");
        rejects_before_input_closes(&["-"], &source);
    }
}

#[test]
fn stdin_dofile_syntax_errors_arrive_before_eof() {
    rejects_before_input_closes(
        &[
            "-e",
            "local ok, err = pcall(dofile); assert(not ok); io.stderr:write(err); os.exit(65)",
        ],
        b"local = 1\n",
    );
}

#[cfg(target_os = "linux")]
#[test]
fn script_files_are_parsed_before_the_next_kernel_read() {
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let directory = Command::new("mktemp")
        .arg("-d")
        .output()
        .expect("create FIFO directory");
    assert!(directory.status.success());
    let directory = PathBuf::from(String::from_utf8(directory.stdout).unwrap().trim());
    let _cleanup = Cleanup(directory.clone());
    let fifo = directory.join("source.lua");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    // Opening both ends keeps this test from blocking on open and keeps EOF
    // away from the interpreter until after it has reported the syntax error.
    let mut writer = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&fifo)
        .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .arg(&fifo)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start FIFO script");
    writer.write_all(b"local = 1\n").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if std::time::Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("file parser waited for EOF after a syntax error");
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    drop(writer);
    assert_eq!(output.status.code(), Some(COMPILE_EXIT), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("expected"));
}

#[test]
fn large_regular_scripts_use_the_pipeline_across_all_profiles() {
    let path = temporary_path("pipeline-profiles");
    let mut source = b"# shell comment\r\nlocal x=0;\n".to_vec();
    for _ in 0..2000 {
        source.extend_from_slice(b"x=x+1;\n");
    }
    source.extend_from_slice(b"local value='\xff'; assert(value==string.char(255)); print(x)\n");
    fs::write(&path, source).unwrap();
    for profile in ["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"] {
        let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat", profile])
            .env_remove("LUA_INIT")
            .env_remove("LUA_INIT_5_5")
            .arg(&path)
            .output()
            .unwrap();
        assert!(output.status.success(), "{profile}: {output:?}");
        assert_eq!(output.stdout, b"2000\n", "{profile}");
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn pipeline_compile_failure_never_executes_the_compiled_prefix() {
    let path = temporary_path("pipeline-compile-failure");
    let source = format!(
        "print('must not run'); local x=0;\n{}local locked<const> =1; locked=2\n",
        "x=x+1;\n".repeat(2000),
    );
    fs::write(&path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args(["--compat", "lua54"])
        .arg(&path)
        .output()
        .unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(output.status.code(), Some(COMPILE_EXIT), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("const"));
}
