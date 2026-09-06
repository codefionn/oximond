//! The CLI runs the interpreter on the main thread when its stack limit can be
//! raised to 64 MiB, and on a spawned 64 MiB thread when it cannot. Both must
//! keep the recursion limits of stock Lua, whatever limit the shell passed in.

use std::io::Write;
use std::process::{Command, Stdio};

const SCRIPT: &str = r#"
local f; f = function() return coroutine.wrap(f)() end
local ok, message = pcall(f)
print(ok, string.find(message, "C stack overflow", 1, true) ~= nil)
local depth = 0
local function recurse() depth = depth + 1; return 1 + recurse() end
local ok, message = pcall(recurse)
print(ok, string.find(message, "stack overflow", 1, true) ~= nil, depth > 100)
local nested = "return " .. string.rep("(", 150) .. "1" .. string.rep(")", 150)
print(load(nested)())
local chunk, message = load("return " .. string.rep("(", 400) .. "1" .. string.rep(")", 400))
print(chunk, string.find(message, "syntax levels", 1, true) ~= nil)
"#;

const EXPECTED: &str = "false	true
false	true	true
1
nil	true
";

/// Runs the binary under `ulimit` arguments and returns its stdout.
fn run_under(limits: &str, execution: &str) -> String {
    let binary = env!("CARGO_BIN_EXE_oximond");
    let output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "{limits}; exec \"$0\" --compat=lua54 --execution={execution} -E -e \"$1\""
        ))
        .arg(binary)
        .arg(SCRIPT)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{limits}: {:?} {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn limits_hold_when_the_stack_limit_can_be_raised() {
    // Soft 8 MiB, hard unlimited: the main thread's stack is grown.
    for execution in ["interpreter", "tracing"] {
        assert_eq!(run_under("ulimit -S -s 8192", execution), EXPECTED);
    }
}

#[test]
fn limits_hold_when_the_hard_limit_forbids_raising_it() {
    // Hard 8 MiB: the interpreter falls back to a thread with its own stack.
    assert_eq!(run_under("ulimit -s 8192", "tracing"), EXPECTED);
}

#[test]
fn limits_hold_when_the_stack_is_already_large() {
    assert_eq!(run_under("ulimit -S -s 131072", "tracing"), EXPECTED);
}

#[test]
fn a_closed_pipe_is_an_error_not_a_signal() {
    // The C entry point replaces the Rust runtime's, which ignores SIGPIPE.
    let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
        .args([
            "-E",
            "-e",
            "io.stdout:setvbuf('no') for i = 1, 100000 do io.write('x\\n') end",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    std::io::stdout().flush().unwrap();
    let status = child.wait().unwrap();
    assert!(status.code().is_some(), "killed by a signal: {status:?}");
}

#[test]
fn shell_children_see_the_stack_limit_the_process_started_with() {
    // The main thread's limit is raised inside the process; os.execute and
    // io.popen children must still report the inherited one, as under stock Lua.
    let binary = env!("CARGO_BIN_EXE_oximond");
    let script = r#"
os.execute("ulimit -S -s")
local pipe = io.popen("ulimit -S -s")
io.write(pipe:read("a"))
pipe:close()
"#;
    let output = Command::new("sh")
        .arg("-c")
        .arg("ulimit -S -s 8192; exec \"$0\" --compat=lua54 -E -e \"$1\"")
        .arg(binary)
        .arg(script)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "8192\n8192\n");
}
