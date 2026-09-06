//! SIGINT becomes a catchable Lua error, and a second signal stops native work.
#![cfg(unix)]

use std::io::{BufRead, BufReader};
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

struct Interpreter {
    child: Child,
    lines: Receiver<String>,
}

impl Interpreter {
    fn start(profile: &str, execution: &str, source: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_oximond"))
            .args(["--compat", profile, "--execution", execution, "-e", source])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start interpreter");
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if sender.send(line.expect("read child output")).is_err() {
                    break;
                }
            }
        });
        Self { child, lines }
    }

    fn line(&self) -> String {
        self.lines
            .recv_timeout(Duration::from_secs(10))
            .expect("child output before deadline")
    }

    fn interrupt(&self) {
        assert!(
            Command::new("kill")
                .args(["-INT", &self.child.id().to_string()])
                .status()
                .expect("send SIGINT")
                .success()
        );
    }

    fn finish(&mut self) -> std::process::ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().expect("wait for child") {
                return status;
            }
            assert!(Instant::now() < deadline, "interpreter did not stop");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn sigint_is_catchable_and_can_be_rearmed() {
    for profile in ["lua51", "lua52", "lua53", "lua54", "lua55", "luajit21"] {
        for execution in ["interpreter", "tracing"] {
            let mut lua = Interpreter::start(
                profile,
                execution,
                r"
                for i = 1, 2 do
                    local ok, err = pcall(function()
                        print('ready')
                        local n = 0
                        while true do n = n + 1 end
                    end)
                    assert(not ok and string.find(err, 'interrupted!', 1, true))
                    print('caught')
                end
                print(42)
            ",
            );
            for _ in 0..2 {
                assert_eq!(lua.line(), "ready", "{profile} {execution}");
                if execution == "tracing" {
                    thread::sleep(Duration::from_millis(100));
                }
                lua.interrupt();
                assert_eq!(lua.line(), "caught", "{profile} {execution}");
            }
            assert_eq!(lua.line(), "42");
            assert!(lua.finish().success(), "{profile} {execution}");
        }
    }
}

#[test]
fn uncaught_sigint_reports_a_runtime_error() {
    let mut lua = Interpreter::start("lua53", "interpreter", "print('ready'); while true do end");
    assert_eq!(lua.line(), "ready");
    lua.interrupt();
    assert_eq!(lua.finish().code(), Some(70));
    let mut stderr = String::new();
    std::io::Read::read_to_string(&mut lua.child.stderr.take().unwrap(), &mut stderr).unwrap();
    assert!(stderr.contains("interrupted!"), "{stderr}");
}

#[test]
fn second_sigint_stops_a_native_pattern_search() {
    // `.*b` alone restarts past the run a failed start covers, which makes
    // the search linear. The `x?` in front keeps it quadratic: an optional
    // atom first disables that restart (`restart_of` in `oximond::pattern`
    // looks only at leading `*` ops), and a match can begin at any byte, so
    // the search still runs long enough for two signals.
    let mut lua = Interpreter::start(
        "lua53",
        "interpreter",
        r"
        local subject = string.rep('a', 100000)
        print('ready')
        string.find(subject, 'x?.*b')
    ",
    );
    assert_eq!(lua.line(), "ready");
    // Give the child time to enter the same long C-style operation as main.lua.
    thread::sleep(Duration::from_secs(1));
    lua.interrupt();
    thread::sleep(Duration::from_millis(50));
    assert!(
        lua.child.try_wait().unwrap().is_none(),
        "first SIGINT stopped native work"
    );
    lua.interrupt();
    assert_eq!(lua.finish().signal(), Some(2));
}

/// A run of calls that never jumps backwards reaches no loop's interrupt
/// check, so the call itself has to look at the flag.
#[test]
fn sigint_stops_a_recursion_that_has_no_loop() {
    for execution in ["interpreter", "tracing"] {
        let mut lua = Interpreter::start(
            "lua54",
            execution,
            r"
            local function spin(n)
                if n == 0 then return 0 end
                return spin(n - 1) + spin(n - 1)
            end
            print('ready')
            local ok, err = pcall(spin, 60)
            assert(not ok and string.find(err, 'interrupted!', 1, true), err)
            print('caught')
            ",
        );
        assert_eq!(lua.line(), "ready", "{execution}");
        thread::sleep(Duration::from_millis(100));
        lua.interrupt();
        assert_eq!(lua.line(), "caught", "{execution}");
        assert!(lua.finish().success(), "{execution}");
    }
}

#[test]
fn sigint_stops_an_endless_tail_call_chain() {
    for execution in ["interpreter", "tracing"] {
        let mut lua = Interpreter::start(
            "lua54",
            execution,
            r"
            local function f(n) return f(n + 1) end
            print('ready')
            local ok, err = pcall(f, 0)
            assert(not ok and string.find(err, 'interrupted!', 1, true), err)
            print('caught')
            ",
        );
        assert_eq!(lua.line(), "ready", "{execution}");
        thread::sleep(Duration::from_millis(100));
        lua.interrupt();
        assert_eq!(lua.line(), "caught", "{execution}");
        assert!(lua.finish().success(), "{execution}");
    }
}
