//! The interactive session behind `-i` and a bare `oximond` on a terminal.
//!
//! It follows stock `lua.c`: a failed line is reported and the session goes
//! on, values a line returns are printed through the global `print`, and a
//! chunk that ends too early asks for more lines under `_PROMPT2`.

use std::io::{self, BufRead, IsTerminal};
use std::sync::atomic::Ordering;

use oximond::bytecode::Chunk;
use oximond::compat::CompatibilityProfile;
use oximond::compiler::{CompileError, CompilerConfig, compile_named};
use oximond::runtime::{RuntimeError, Value};
use oximond::vm::{Vm, VmError};

use crate::INTERRUPT_FLAG;

/// Turns an error object into the text the session prints: the message and
/// a traceback. It keeps its own references to the library functions, so a
/// session that overwrites `debug` or `tostring` still gets its errors.
const MESSAGE_HANDLER: &str = "\
local type, tostring = type, tostring
local traceback = debug and debug.traceback
local getmetatable = debug and debug.getmetatable or getmetatable
return function(e)
  if type(e) ~= 'string' then
    local mt = getmetatable(e)
    local text = mt and mt.__tostring and tostring(e)
    if type(text) == 'string' then return text end
    if type(e) ~= 'number' then
      e = '(error object is a ' .. type(e) .. ' value)'
    end
  end
  if traceback then return traceback(e, 2) end
  return e
end";

enum Line {
    Text(Vec<u8>),
    /// Ctrl-C at the prompt: drop the input typed so far.
    Cancelled,
    End,
}

/// Where lines come from: the line editor on a terminal, plain reads from
/// standard input otherwise.
enum Input {
    Editor(Box<rustyline::DefaultEditor>),
    Plain,
}

impl Input {
    fn open() -> Self {
        if io::stdin().is_terminal()
            && io::stdout().is_terminal()
            && let Ok(editor) = rustyline::DefaultEditor::new()
        {
            return Self::Editor(Box::new(editor));
        }
        Self::Plain
    }

    fn read(&mut self, vm: &mut Vm, prompt: &[u8]) -> Result<Line, VmError> {
        match self {
            Self::Editor(editor) => {
                // The editor draws the prompt itself, straight to the terminal.
                vm.runtime_mut().flush_output()?;
                let prompt = String::from_utf8_lossy(prompt);
                match editor.readline(prompt.as_ref()) {
                    Ok(line) => {
                        if !line.trim().is_empty() {
                            let _ = editor.add_history_entry(line.as_str());
                        }
                        Ok(Line::Text(line.into_bytes()))
                    }
                    Err(rustyline::error::ReadlineError::Interrupted) => Ok(Line::Cancelled),
                    Err(rustyline::error::ReadlineError::Eof) => Ok(Line::End),
                    Err(error) => Err(read_error(&error)),
                }
            }
            Self::Plain => {
                vm.runtime_mut().write_output(prompt)?;
                vm.runtime_mut().flush_output()?;
                // The lock is released before the line runs, so the line's
                // own `io.read` can take standard input.
                let mut line = Vec::new();
                let read = io::stdin()
                    .lock()
                    .read_until(b'\n', &mut line)
                    .map_err(|error| read_error(&error))?;
                if read == 0 {
                    return Ok(Line::End);
                }
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                Ok(Line::Text(line))
            }
        }
    }
}

fn read_error(error: &dyn std::fmt::Display) -> VmError {
    RuntimeError::message(format!("cannot read standard input: {error}")).into()
}

/// True when standard input is a terminal, where a bare `oximond` starts a
/// session instead of reading a script.
pub fn stdin_is_terminal() -> bool {
    io::stdin().is_terminal()
}

/// Runs the session until standard input ends.
///
/// # Errors
///
/// Returns an `os.exit` request, a failed write to the output, or a failed
/// read of standard input. Errors of the lines themselves are printed.
pub fn run(vm: &mut Vm) -> Result<Vec<Value>, VmError> {
    let profile = vm.compatibility_profile();
    let config = CompilerConfig::new(profile);
    // Lua 5.3 and later print the value of a line that is an expression.
    let implicit_return = matches!(
        profile,
        CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    );
    let handler = compile_named("=(message handler)", MESSAGE_HANDLER, config)
        .map_err(|error| RuntimeError::message(error.to_string()))?;
    let handler = vm
        .execute(&handler)?
        .into_iter()
        .next()
        .unwrap_or(Value::Nil);
    let mut input = Input::open();
    if matches!(input, Input::Editor(_)) {
        let banner = format!("Oximond {} ({profile})\n", env!("CARGO_PKG_VERSION"));
        vm.runtime_mut().write_output(banner.as_bytes())?;
    }

    let mut pending = Vec::new();
    let mut expression = false;
    loop {
        let prompt = prompt(vm, !pending.is_empty());
        let line = match input.read(vm, &prompt)? {
            Line::Text(line) => line,
            Line::Cancelled => {
                pending.clear();
                expression = false;
                continue;
            }
            Line::End => break,
        };
        if pending.is_empty() {
            let trimmed = line.trim_ascii();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix(b"=") {
                // `=expression`, the spelling every Lua version accepts.
                pending.extend_from_slice(rest);
                expression = true;
            } else {
                if implicit_return
                    && let Ok(chunk) = compile_named("=stdin", returning(&line), config)
                {
                    run_chunk(vm, &chunk, &handler)?;
                    continue;
                }
                pending.extend_from_slice(&line);
            }
        } else {
            pending.extend_from_slice(&line);
        }
        pending.push(b'\n');
        match compile_pending(&pending, expression, config) {
            Ok(chunk) => run_chunk(vm, &chunk, &handler)?,
            Err(error) if is_incomplete(&error, &pending) => continue,
            Err(error) => report(vm, &error.to_string())?,
        }
        pending.clear();
        expression = false;
    }
    if !pending.is_empty() {
        // Input ended inside a chunk. Run it if it is whole, as a last line
        // without a newline is, and report it otherwise.
        match compile_pending(&pending, expression, config) {
            Ok(chunk) => run_chunk(vm, &chunk, &handler)?,
            Err(error) => report(vm, &error.to_string())?,
        }
    }
    // The editor has already moved to a fresh line on Ctrl-D.
    if matches!(input, Input::Plain) {
        vm.runtime_mut().write_output(b"\n")?;
    }
    Ok(Vec::new())
}

fn returning(source: &[u8]) -> Vec<u8> {
    let mut candidate = Vec::with_capacity(source.len() + 8);
    candidate.extend_from_slice(b"return ");
    candidate.extend_from_slice(source);
    // On its own line, so a comment that ends the source cannot hide it.
    candidate.push(b'\n');
    candidate
}

fn compile_pending(
    pending: &[u8],
    expression: bool,
    config: CompilerConfig,
) -> Result<Chunk, CompileError> {
    if expression {
        compile_named("=stdin", returning(pending), config)
    } else {
        compile_named("=stdin", pending, config)
    }
}

/// Runs one chunk of the session and prints the values it returns.
fn run_chunk(vm: &mut Vm, chunk: &Chunk, handler: &Value) -> Result<(), VmError> {
    // A Ctrl-C that arrived while the session waited for input is not meant
    // for this chunk.
    if let Some(flag) = INTERRUPT_FLAG.get() {
        flag.store(false, Ordering::Relaxed);
    }
    let values = match vm.execute_with_message_handler(chunk, Vec::new(), handler.clone()) {
        Ok(values) => values,
        Err(error) => return report_failure(vm, error, None),
    };
    if values.is_empty() {
        return Ok(());
    }
    let print = vm.runtime().get_global("print");
    match vm.call_value(print, values) {
        Ok(_) => Ok(()),
        Err(error) => report_failure(vm, error, Some("error calling 'print'")),
    }
}

/// Prints the error of a line, or passes on one that ends the session.
fn report_failure(vm: &mut Vm, error: VmError, context: Option<&str>) -> Result<(), VmError> {
    let ends_session = error.exit_request().is_some()
        || matches!(
            &error,
            VmError::Runtime(RuntimeError::HostOutputIo { .. })
                | VmError::LocatedRuntime {
                    error: RuntimeError::HostOutputIo { .. },
                    ..
                }
        );
    if ends_session {
        return Err(error);
    }
    match context {
        Some(context) => report(vm, &format!("{context} ({error})")),
        None => report(vm, without_host_frames(&error.to_string())),
    }
}

/// Cuts the traceback where the chunk was entered. The frames below belong
/// to the protected call that ran it, and to nothing the user typed.
fn without_host_frames(message: &str) -> &str {
    message
        .rfind("\n\t[C]: in function 'xpcall'")
        .map_or(message, |entry| &message[..entry])
}

fn report(vm: &mut Vm, message: &str) -> Result<(), VmError> {
    // Output the line wrote before it failed comes first.
    vm.runtime_mut().flush_output()?;
    let runtime = vm.runtime_mut();
    runtime.write_error_output(message.as_bytes())?;
    runtime.write_error_output(b"\n")?;
    runtime.flush_error_output()?;
    Ok(())
}

fn prompt(vm: &Vm, continuation: bool) -> Vec<u8> {
    let name = if continuation { "_PROMPT2" } else { "_PROMPT" };
    match vm.runtime().get_global(name) {
        Value::String(prompt) => prompt.as_bytes().to_vec(),
        _ if continuation => b">> ".to_vec(),
        _ => b"> ".to_vec(),
    }
}

/// True when `source` failed to compile only because it stops too early, so
/// another line may complete it.
fn is_incomplete(error: &CompileError, source: &[u8]) -> bool {
    let CompileError::Syntax(error) = error else {
        return false;
    };
    error.message.starts_with("unterminated long")
        || error.span.start >= source.len().saturating_sub(1)
        || source.ends_with(b"\\\n")
        || (error.message.starts_with("expected assignment") && source.ends_with(b" \n"))
}
