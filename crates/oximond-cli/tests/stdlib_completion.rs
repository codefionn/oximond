use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn debug_console_executes_commands_reports_errors_and_returns_on_cont() {
    for profile in ["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"] {
        for execution in ["interpreter", "tracing"] {
            let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
                .args([
                    "--compat",
                    profile,
                    "--execution",
                    execution,
                    "-e",
                    "debug.debug(); print(_G.console_answer)",
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(
                    b"local =\nerror('console failure')\nrawset(_G,'console_answer',42)\ncont\n",
                )
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(output.status.success(), "{profile}/{execution}: {output:?}");
            assert_eq!(output.stdout, b"42\n", "{profile}/{execution}");
            let errors = String::from_utf8_lossy(&output.stderr);
            assert!(errors.contains("console failure"), "{errors}");
            assert!(errors.contains("lua_debug> "), "{errors}");
        }
    }
}

#[cfg(unix)]
#[test]
fn popen_reads_and_returns_process_exit_status_per_profile() {
    for profile in ["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"] {
        for execution in ["interpreter", "tracing"] {
            let output = Command::new(env!("CARGO_BIN_EXE_oximond"))
                .args([
                    "--compat",
                    profile,
                    "--execution",
                    execution,
                    "-e",
                    r#"
                    local f = assert(io.popen("printf 'pipe-output'; exit 7",'r'))
                    print(f:read('*a'))
                    local ok, kind, code = f:close()
                    print(ok, kind, code)
                "#,
                ])
                .output()
                .unwrap();
            assert!(output.status.success(), "{profile}/{execution}: {output:?}");
            let expected = if profile == "lua51" || profile == "luajit21" {
                "pipe-output\ntrue\tnil\tnil\n"
            } else {
                "pipe-output\nnil\texit\t7\n"
            };
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                expected,
                "{profile}/{execution}"
            );
        }
    }
}
