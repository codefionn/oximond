//! Runs one interpreter process under a deadline, a memory cap and an
//! output cap.

use std::ffi::OsString;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Resource limits for one process.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub timeout: Duration,
    /// Address-space cap applied through `prlimit --as`. Zero disables it.
    pub memory_mib: u64,
    /// Bytes kept from each of stdout and stderr. A process that writes
    /// more is killed.
    pub output_cap: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(20),
            memory_mib: 2048,
            output_cap: 8 << 20,
        }
    }
}

/// How a process ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Exited(i32),
    /// Killed by a signal it did not ask for (a crash, or the memory cap).
    Signaled(i32),
    TimedOut,
    OutputCapExceeded,
    SpawnFailed(String),
}

impl Status {
    #[must_use]
    pub const fn success(&self) -> bool {
        matches!(self, Self::Exited(0))
    }

    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Exited(code) => format!("exit {code}"),
            Self::Signaled(signal) => format!("signal {signal}"),
            Self::TimedOut => "timeout".to_owned(),
            Self::OutputCapExceeded => "output cap exceeded".to_owned(),
            Self::SpawnFailed(error) => format!("spawn failed: {error}"),
        }
    }
}

/// What one process produced.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub status: Status,
    pub elapsed: Duration,
}

/// The full command line `run` executes, for reports.
#[must_use]
pub fn command_line(program: &[OsString], limits: Limits) -> Vec<OsString> {
    let mut line = Vec::new();
    if limits.memory_mib > 0 {
        line.push(OsString::from("prlimit"));
        line.push(OsString::from(format!("--as={}", limits.memory_mib << 20)));
        line.push(OsString::from("--"));
    }
    line.extend(program.iter().cloned());
    line
}

/// Runs `program` (command and arguments) in `directory` with extra
/// environment variables, in its own process group so a timeout kills
/// everything it started.
///
/// # Panics
///
/// Panics if the child's stdout or stderr was not piped, which `run` itself
/// sets up.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "spawn, poll and collect in one place"
)]
pub fn run(
    program: &[OsString],
    directory: &Path,
    environment: &[(&str, &str)],
    limits: Limits,
) -> Outcome {
    use std::os::unix::process::{CommandExt, ExitStatusExt};

    let line = command_line(program, limits);
    let started = Instant::now();
    let mut command = Command::new(&line[0]);
    command
        .args(&line[1..])
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    for (key, value) in environment {
        command.env(key, value);
    }
    // ETXTBSY: an executable written moments ago can still be open for
    // writing in a child another thread forked before its exec. It clears
    // once that child execs.
    let mut spawned = command.spawn();
    for _ in 0..50 {
        match &spawned {
            Err(error) if error.raw_os_error() == Some(26) => {
                thread::sleep(Duration::from_millis(10));
                spawned = command.spawn();
            }
            _ => break,
        }
    }
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            return Outcome {
                stdout: Vec::new(),
                stderr: Vec::new(),
                status: Status::SpawnFailed(format!("{}: {error}", line[0].to_string_lossy())),
                elapsed: started.elapsed(),
            };
        }
    };
    let overflow = Arc::new(AtomicBool::new(false));
    let stdout = capture(
        child.stdout.take().expect("stdout is piped"),
        limits.output_cap,
        &overflow,
    );
    let stderr = capture(
        child.stderr.take().expect("stderr is piped"),
        limits.output_cap,
        &overflow,
    );
    let mut status = None;
    while status.is_none() {
        match child.try_wait() {
            Ok(Some(exit)) => {
                status = Some(match (exit.code(), exit.signal()) {
                    (Some(code), _) => Status::Exited(code),
                    (None, Some(signal)) => Status::Signaled(signal),
                    (None, None) => Status::Exited(-1),
                });
            }
            Ok(None) => {
                if overflow.load(Ordering::Relaxed) {
                    kill_group(child.id());
                    let _ = child.wait();
                    status = Some(Status::OutputCapExceeded);
                } else if started.elapsed() > limits.timeout {
                    kill_group(child.id());
                    let _ = child.wait();
                    status = Some(Status::TimedOut);
                } else {
                    thread::sleep(Duration::from_millis(2));
                }
            }
            Err(error) => {
                kill_group(child.id());
                status = Some(Status::SpawnFailed(error.to_string()));
            }
        }
    }
    // A grandchild could hold the pipes open; the group kill above covers
    // timeouts, and a normal exit closes them.
    kill_group(child.id());
    let stdout = stdout.join().unwrap_or_default();
    let stderr = stderr.join().unwrap_or_default();
    let mut status = status.expect("the loop ends with a status");
    // The capture thread stops reading at the cap, so the process may die
    // of SIGPIPE before the loop above sees the flag.
    if overflow.load(Ordering::Relaxed) {
        status = Status::OutputCapExceeded;
    }
    Outcome {
        stdout,
        stderr,
        status,
        elapsed: started.elapsed(),
    }
}

fn capture(
    mut stream: impl Read + Send + 'static,
    cap: usize,
    overflow: &Arc<AtomicBool>,
) -> thread::JoinHandle<Vec<u8>> {
    let overflow = Arc::clone(overflow);
    thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            match stream.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    let room = cap.saturating_sub(kept.len());
                    kept.extend_from_slice(&buffer[..read.min(room)]);
                    if read > room {
                        overflow.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }
        kept
    })
}

/// Kills the process group `leader` leads. The group exists because the
/// child was spawned with `process_group(0)`.
fn kill_group(leader: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", &format!("-{leader}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(test)]
mod tests {
    use super::{Limits, Status, run};
    use std::ffi::OsString;
    use std::path::Path;
    use std::time::Duration;

    fn shell(script: &str) -> Vec<OsString> {
        vec!["sh".into(), "-c".into(), script.into()]
    }

    #[test]
    fn captures_output_and_exit_status() {
        let outcome = run(
            &shell("printf 'a\\0b'; echo err >&2; exit 3"),
            Path::new("."),
            &[],
            Limits::default(),
        );
        assert_eq!(outcome.stdout, b"a\0b");
        assert_eq!(outcome.stderr, b"err\n");
        assert_eq!(outcome.status, Status::Exited(3));
    }

    #[test]
    fn kills_a_process_that_runs_past_the_deadline() {
        let limits = Limits {
            timeout: Duration::from_millis(200),
            ..Limits::default()
        };
        let outcome = run(
            &shell("sleep 30 & sleep 30; echo late"),
            Path::new("."),
            &[],
            limits,
        );
        assert_eq!(outcome.status, Status::TimedOut);
        assert!(outcome.elapsed < Duration::from_secs(10));
    }

    #[test]
    fn kills_a_process_that_floods_its_output() {
        let limits = Limits {
            output_cap: 1000,
            ..Limits::default()
        };
        let outcome = run(&shell("yes"), Path::new("."), &[], limits);
        assert_eq!(outcome.status, Status::OutputCapExceeded);
        assert_eq!(outcome.stdout.len(), 1000);
    }

    #[test]
    fn reports_a_missing_program() {
        let limits = Limits {
            memory_mib: 0,
            ..Limits::default()
        };
        let outcome = run(&["/nonexistent/lua".into()], Path::new("."), &[], limits);
        assert!(matches!(outcome.status, Status::SpawnFailed(_)));
    }
}
