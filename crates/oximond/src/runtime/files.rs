//! Runtime-owned file objects and the Lua `io` file operations.

use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::gc::GcReference;
use crate::runtime::helpers::{
    file_io_error, file_line_iterator_next, file_offset, file_stream_error, file_write_bytes,
    flush_pending, parse_lua_integer_exact, parse_lua_number, usize_to_lua_number,
    write_named_file,
};
use crate::runtime::number_format::NumberFormat;
use crate::runtime::runtime::Runtime;
use crate::runtime::value::{BuiltinFunction, FileHandle, Value};
use crate::string::LuaString;
use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::io::Write;
use std::io::{BufRead, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

pub(crate) struct RuntimeFile {
    pub(crate) storage: FileStorage,
    pub(crate) uservalue: Value,
}

pub(crate) enum FileStorage {
    Memory {
        bytes: Vec<u8>,
        cursor: usize,
    },
    Named {
        file: File,
        readable: bool,
        writable: bool,
        buffering: FileBuffering,
        pending: Vec<u8>,
    },
    PipeRead {
        child: Child,
        output: BufReader<ChildStdout>,
    },
    PipeWrite {
        child: Child,
        input: ChildStdin,
    },
    HostInput,
    HostOutput,
    HostError,
    Closed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FileBuffering {
    No,
    Full(usize),
    Line(usize),
}

impl Drop for RuntimeFile {
    fn drop(&mut self) {
        if let FileStorage::Named { file, pending, .. } = &mut self.storage {
            let _ = file.write_all(pending);
            let _ = file.flush();
        }
        if matches!(
            self.storage,
            FileStorage::PipeRead { .. } | FileStorage::PipeWrite { .. }
        ) {
            let storage = std::mem::replace(&mut self.storage, FileStorage::Closed);
            match storage {
                FileStorage::PipeRead { mut child, output } => {
                    drop(output);
                    let _ = child.wait();
                }
                FileStorage::PipeWrite { mut child, input } => {
                    drop(input);
                    let _ = child.wait();
                }
                _ => unreachable!(),
            }
        }
    }
}

/// How `io.lines` and `file:lines` treat read formats after their file
/// argument, following the Lua release a profile targets.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LinesFormats {
    /// Lua 5.1 and LuaJIT: extra arguments are ignored and every call reads
    /// one line.
    #[default]
    Ignored,
    /// Lua 5.2: up to 17 formats are passed to `read` on each call.
    Lua52,
    /// Lua 5.3: up to 250 formats are passed to `read` on each call.
    Lua53,
    /// Lua 5.4+: as Lua 5.3, and `io.lines(name)` also returns its file as
    /// the loop's to-be-closed value.
    Lua54,
}

impl LinesFormats {
    fn check_count(
        self,
        count: usize,
        function: &str,
        first_format_argument: usize,
    ) -> RuntimeResult<()> {
        // Stock `aux_lines` counts formats against a fixed stack budget and
        // reports a fixed argument position, shifted by one for methods.
        let (limit, reported, reason) = match self {
            Self::Ignored => return Ok(()),
            Self::Lua52 => (17, first_format_argument + 15, "too many options"),
            Self::Lua53 | Self::Lua54 => (250, first_format_argument + 250, "too many arguments"),
        };
        if count > limit {
            return Err(RuntimeError::message(format!(
                "bad argument #{reported} to '{function}' ({reason})"
            )));
        }
        Ok(())
    }
}

/// Opening modes supported by runtime-owned named files.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileOpenMode {
    Read,
    Write,
    Append,
    ReadUpdate,
    WriteUpdate,
    AppendUpdate,
}

impl FileOpenMode {
    /// Parses the Lua 5.1 file modes accepted by `io.open`. A trailing `b`
    /// is accepted and has no effect on platforms without text translation.
    ///
    /// # Errors
    ///
    /// Returns an error when `mode` is not a supported file mode.
    pub fn parse(mode: &[u8]) -> RuntimeResult<Self> {
        let mode = mode.strip_suffix(b"b").unwrap_or(mode);
        match mode {
            b"r" => Ok(Self::Read),
            b"w" => Ok(Self::Write),
            b"a" => Ok(Self::Append),
            b"r+" => Ok(Self::ReadUpdate),
            b"w+" => Ok(Self::WriteUpdate),
            b"a+" => Ok(Self::AppendUpdate),
            _ => Err(RuntimeError::message("invalid file mode")),
        }
    }

    fn open_options(self) -> OpenOptions {
        let mut options = OpenOptions::new();
        match self {
            Self::Read => {
                options.read(true);
            }
            Self::Write => {
                options.write(true).create(true).truncate(true);
            }
            Self::Append => {
                options.append(true).create(true);
            }
            Self::ReadUpdate => {
                options.read(true).write(true);
            }
            Self::WriteUpdate => {
                options.read(true).write(true).create(true).truncate(true);
            }
            Self::AppendUpdate => {
                options.read(true).append(true).create(true);
            }
        }
        options
    }

    const fn readable(self) -> bool {
        matches!(
            self,
            Self::Read | Self::ReadUpdate | Self::WriteUpdate | Self::AppendUpdate
        )
    }

    const fn writable(self) -> bool {
        !matches!(self, Self::Read)
    }
}

impl Runtime {
    /// Starts a shell command with a readable or writable process pipe.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the shell cannot be started.
    pub fn open_process(
        &mut self,
        command: impl AsRef<std::ffi::OsStr>,
        read: bool,
    ) -> RuntimeResult<FileHandle> {
        let mut process = if cfg!(windows) {
            Command::new("cmd")
        } else {
            Command::new("sh")
        };
        process.arg(if cfg!(windows) { "/C" } else { "-c" });
        process.arg(command);
        crate::runtime::child_limits::restore_in_child(&mut process);
        let storage = if read {
            let mut child = process
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|error| file_io_error(&error))?;
            let output = child
                .stdout
                .take()
                .ok_or_else(|| RuntimeError::message("process stdout pipe is unavailable"))?;
            FileStorage::PipeRead {
                child,
                output: BufReader::new(output),
            }
        } else {
            let mut child = process
                .stdin(Stdio::piped())
                .spawn()
                .map_err(|error| file_io_error(&error))?;
            let input = child
                .stdin
                .take()
                .ok_or_else(|| RuntimeError::message("process stdin pipe is unavailable"))?;
            FileStorage::PipeWrite { child, input }
        };
        self.create_host_file(storage, "process pipe")
    }
    /// Creates an opaque, byte-oriented in-memory temporary file object.
    ///
    /// # Errors
    ///
    /// Returns an error if the file arena cannot allocate another handle.
    pub fn create_temp_file(&mut self) -> RuntimeResult<FileHandle> {
        let handle = self
            .files
            .insert(RuntimeFile {
                storage: FileStorage::Memory {
                    bytes: Vec::new(),
                    cursor: 0,
                },
                uservalue: Value::Nil,
            })
            .map(FileHandle)
            .ok_or_else(|| RuntimeError::message("temporary file handle space exhausted"))?;
        self.gc_charge(std::mem::size_of::<RuntimeFile>());
        Ok(handle)
    }

    /// Creates the opaque file object used for the host's standard-error stream.
    ///
    /// Its writes use the runtime output sink so embedders can route output
    /// without exposing a platform file descriptor to Lua.
    ///
    /// # Errors
    ///
    /// Returns an error if the file arena cannot allocate another handle.
    pub fn create_stderr_file(&mut self) -> RuntimeResult<FileHandle> {
        let handle = self
            .files
            .insert(RuntimeFile {
                storage: FileStorage::HostError,
                uservalue: Value::Nil,
            })
            .map(FileHandle)
            .ok_or_else(|| RuntimeError::message("standard error handle space exhausted"))?;
        self.gc_charge(std::mem::size_of::<RuntimeFile>());
        Ok(handle)
    }

    /// Creates the opaque file object used for the host's standard input.
    ///
    /// # Errors
    ///
    /// Returns an error if the file arena has exhausted its handle space.
    pub fn create_stdin_file(&mut self) -> RuntimeResult<FileHandle> {
        self.create_host_file(FileStorage::HostInput, "standard input")
    }

    /// Creates the opaque file object used for the host's standard output.
    ///
    /// # Errors
    ///
    /// Returns an error if the file arena has exhausted its handle space.
    pub fn create_stdout_file(&mut self) -> RuntimeResult<FileHandle> {
        self.create_host_file(FileStorage::HostOutput, "standard output")
    }

    pub(crate) fn create_host_file(
        &mut self,
        storage: FileStorage,
        description: &'static str,
    ) -> RuntimeResult<FileHandle> {
        let handle = self
            .files
            .insert(RuntimeFile {
                storage,
                uservalue: Value::Nil,
            })
            .map(FileHandle)
            .ok_or_else(|| {
                RuntimeError::message(format!("{description} handle space exhausted"))
            })?;
        self.gc_charge(std::mem::size_of::<RuntimeFile>());
        Ok(handle)
    }

    /// Opens a named file without exposing the operating-system handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or the file arena is
    /// out of handle space.
    pub fn open_file(
        &mut self,
        path: impl AsRef<Path>,
        mode: FileOpenMode,
    ) -> RuntimeResult<FileHandle> {
        let file = mode
            .open_options()
            .open(path)
            .map_err(|error| file_io_error(&error))?;
        let handle = self
            .files
            .insert(RuntimeFile {
                storage: FileStorage::Named {
                    file,
                    readable: mode.readable(),
                    writable: mode.writable(),
                    buffering: FileBuffering::No,
                    pending: Vec::new(),
                },
                uservalue: Value::Nil,
            })
            .map(FileHandle)
            .ok_or_else(|| RuntimeError::message("file handle space exhausted"))?;
        self.gc_charge(std::mem::size_of::<RuntimeFile>());
        Ok(handle)
    }

    /// User value of an I/O file handle in the Lua 5.2 and 5.3 profiles.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale.
    pub fn file_uservalue(&self, handle: FileHandle) -> RuntimeResult<Value> {
        Ok(self.file(handle)?.uservalue.clone())
    }

    /// Updates the file's GC edge. Closed file handles retain their user value.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale.
    pub fn set_file_uservalue(&mut self, handle: FileHandle, value: Value) -> RuntimeResult<()> {
        self.file(handle)?;
        self.gc_barrier_value(Some(GcReference::File(handle)), &value);
        self.files
            .get_mut(handle.0)
            .ok_or(RuntimeError::InvalidFile(handle))?
            .uservalue = value;
        Ok(())
    }

    /// Returns the current default input handle, if one was selected.
    #[must_use]
    pub const fn default_input(&self) -> Option<FileHandle> {
        self.default_input
    }

    /// Selects a default input handle. `None` restores the host input stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is invalid or closed.
    pub fn set_default_input(&mut self, handle: Option<FileHandle>) -> RuntimeResult<()> {
        if let Some(handle) = handle {
            self.file_open(handle)?;
        }
        self.default_input = handle;
        Ok(())
    }

    /// Returns the current default output handle, if one was selected.
    #[must_use]
    pub const fn default_output(&self) -> Option<FileHandle> {
        self.default_output
    }

    /// Selects a default output handle. `None` restores runtime host output.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is invalid or closed.
    pub fn set_default_output(&mut self, handle: Option<FileHandle>) -> RuntimeResult<()> {
        if let Some(handle) = handle {
            self.file_open(handle)?;
        }
        self.default_output = handle;
        Ok(())
    }

    /// Resolves a supported file method without exposing file storage.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is stale or if `key` is not a method name.
    pub fn file_method(&self, handle: FileHandle, key: &Value) -> RuntimeResult<Value> {
        self.file(handle)?;
        let Value::String(key) = key else {
            return Ok(Value::Nil);
        };
        Ok(match key.as_bytes() {
            b"write" => Value::BuiltinFunction(BuiltinFunction::FileWrite),
            b"seek" => Value::BuiltinFunction(BuiltinFunction::FileSeek),
            b"read" => Value::BuiltinFunction(BuiltinFunction::FileRead),
            b"close" => Value::BuiltinFunction(BuiltinFunction::FileClose),
            b"flush" => Value::BuiltinFunction(BuiltinFunction::FileFlush),
            b"setvbuf" => Value::BuiltinFunction(BuiltinFunction::FileSetVbuf),
            b"lines" => Value::BuiltinFunction(BuiltinFunction::FileLines),
            _ => Value::Nil,
        })
    }

    /// Writes byte-preserving values at the current file cursor.
    ///
    /// # Errors
    ///
    /// Returns an error for stale, closed, or non-string/non-number file input.
    pub fn file_write(&mut self, handle: FileHandle, values: &[Value]) -> RuntimeResult<()> {
        let format = self.number_format.for_file_write();
        if matches!(&self.file(handle)?.storage, FileStorage::HostOutput) {
            for value in values {
                self.write_output(&file_write_bytes(value, format)?)?;
            }
            return Ok(());
        }
        if matches!(&self.file(handle)?.storage, FileStorage::HostError) {
            for value in values {
                self.write_error_output(&file_write_bytes(value, format)?)?;
            }
            return Ok(());
        }
        let file = self.file_mut_open(handle)?;
        for value in values {
            let bytes = file_write_bytes(value, format)?;
            match file {
                FileStorage::Memory {
                    bytes: data,
                    cursor,
                } => {
                    let end = cursor
                        .checked_add(bytes.len())
                        .ok_or_else(|| RuntimeError::message("temporary file is too large"))?;
                    if end > data.len() {
                        data.resize(end, 0);
                    }
                    data[*cursor..end].copy_from_slice(&bytes);
                    *cursor = end;
                }
                FileStorage::Named {
                    file,
                    writable,
                    buffering,
                    pending,
                    ..
                } => {
                    if !*writable {
                        return Err(file_stream_error("Bad file descriptor"));
                    }
                    write_named_file(file, *buffering, pending, &bytes)?;
                }
                FileStorage::PipeWrite { input, .. } => input
                    .write_all(&bytes)
                    .map_err(|error| file_io_error(&error))?,
                FileStorage::PipeRead { .. } => {
                    return Err(file_stream_error("Bad file descriptor"));
                }
                FileStorage::HostInput => {
                    return Err(file_stream_error("file is not writable"));
                }
                FileStorage::HostOutput | FileStorage::HostError => {
                    unreachable!("host output is handled before mutation")
                }
                FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
            }
        }
        Ok(())
    }

    /// Moves a file cursor and returns its zero-based byte offset.
    ///
    /// # Errors
    ///
    /// Returns an error for stale, closed, malformed, or out-of-range seeks.
    pub fn file_seek(
        &mut self,
        handle: FileHandle,
        whence: &Value,
        offset: &Value,
    ) -> RuntimeResult<f64> {
        let file = self.file_mut_open(handle)?;
        let Value::String(whence) = whence else {
            return Err(RuntimeError::TypeError {
                operation: "seek",
                expected: "string",
                found: whence.type_name(),
            });
        };
        let offset = file_offset(offset)?;
        match file {
            FileStorage::Memory { bytes, cursor } => {
                let base = match whence.as_bytes() {
                    b"set" => 0_i128,
                    b"cur" => i128::try_from(*cursor)
                        .map_err(|_| RuntimeError::message("temporary file offset is too large"))?,
                    b"end" => i128::try_from(bytes.len())
                        .map_err(|_| RuntimeError::message("temporary file offset is too large"))?,
                    _ => return Err(RuntimeError::message("invalid seek mode")),
                };
                let target = base
                    .checked_add(i128::from(offset))
                    .ok_or_else(|| RuntimeError::message("invalid seek position"))?;
                let target = usize::try_from(target)
                    .map_err(|_| RuntimeError::message("invalid seek position"))?;
                *cursor = target;
                usize_to_lua_number(target, "temporary file offset is too large")
            }
            FileStorage::Named { file, pending, .. } => {
                flush_pending(file, pending)?;
                let position = match whence.as_bytes() {
                    b"set" => {
                        let offset = u64::try_from(offset)
                            .map_err(|_| RuntimeError::message("invalid seek position"))?;
                        SeekFrom::Start(offset)
                    }
                    b"cur" => SeekFrom::Current(offset),
                    b"end" => SeekFrom::End(offset),
                    _ => return Err(RuntimeError::message("invalid seek mode")),
                };
                let position = file.seek(position).map_err(|error| file_io_error(&error))?;
                position
                    .to_string()
                    .parse::<f64>()
                    .map_err(|_| RuntimeError::message("file offset is too large"))
            }
            FileStorage::HostInput | FileStorage::HostOutput | FileStorage::HostError => {
                Err(file_stream_error("file is not seekable"))
            }
            FileStorage::PipeRead { .. } | FileStorage::PipeWrite { .. } => {
                Err(file_stream_error("Illegal seek"))
            }
            FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
        }
    }

    /// Reads all remaining bytes from a file.
    ///
    /// # Errors
    ///
    /// Returns an error for stale or closed file handles.
    pub fn file_read_all(&mut self, handle: FileHandle) -> RuntimeResult<LuaString> {
        let file = self.file_mut_open(handle)?;
        match file {
            FileStorage::Memory { bytes, cursor } => {
                let result = LuaString::from(bytes[(*cursor).min(bytes.len())..].to_vec());
                *cursor = bytes.len();
                Ok(result)
            }
            FileStorage::Named {
                file,
                readable,
                pending,
                ..
            } => {
                if !*readable {
                    return Ok(LuaString::from(Vec::new()));
                }
                flush_pending(file, pending)?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)
                    .map_err(|error| file_io_error(&error))?;
                Ok(LuaString::from(bytes))
            }
            FileStorage::HostInput => {
                let mut bytes = Vec::new();
                std::io::stdin()
                    .read_to_end(&mut bytes)
                    .map_err(|error| file_io_error(&error))?;
                Ok(LuaString::from(bytes))
            }
            FileStorage::PipeRead { output, .. } => {
                let mut bytes = Vec::new();
                output
                    .read_to_end(&mut bytes)
                    .map_err(|error| file_io_error(&error))?;
                Ok(LuaString::from(bytes))
            }
            FileStorage::PipeWrite { .. } => Err(file_stream_error("Bad file descriptor")),
            FileStorage::HostOutput | FileStorage::HostError => {
                Err(file_stream_error("file is not readable"))
            }
            FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
        }
    }

    /// Reads up to `count` bytes, returning `None` only at end of file.
    ///
    /// # Errors
    ///
    /// Returns an error for stale or closed file handles, or an I/O error for
    /// named files.
    pub fn file_read_some(
        &mut self,
        handle: FileHandle,
        count: usize,
    ) -> RuntimeResult<Option<LuaString>> {
        if count == 0 {
            let available = self.file_peek_byte(handle, |_| false)?.is_some();
            return Ok(available.then(|| LuaString::from(Vec::new())));
        }
        let file = self.file_mut_open(handle)?;
        match file {
            FileStorage::Memory { bytes, cursor } => {
                if *cursor >= bytes.len() {
                    return Ok(None);
                }
                let end = cursor.saturating_add(count).min(bytes.len());
                let result = LuaString::from(bytes[*cursor..end].to_vec());
                *cursor = end;
                Ok(Some(result))
            }
            FileStorage::Named {
                file,
                readable,
                pending,
                ..
            } => {
                if !*readable {
                    return Ok(None);
                }
                flush_pending(file, pending)?;
                let mut bytes = vec![0; count];
                let read = file
                    .read(&mut bytes)
                    .map_err(|error| file_io_error(&error))?;
                if read == 0 {
                    Ok(None)
                } else {
                    bytes.truncate(read);
                    Ok(Some(LuaString::from(bytes)))
                }
            }
            FileStorage::HostInput => {
                let mut bytes = vec![0; count];
                let read = std::io::stdin()
                    .read(&mut bytes)
                    .map_err(|error| file_io_error(&error))?;
                if read == 0 {
                    Ok(None)
                } else {
                    bytes.truncate(read);
                    Ok(Some(LuaString::from(bytes)))
                }
            }
            FileStorage::PipeRead { output, .. } => {
                let mut bytes = vec![0; count];
                let read = output
                    .read(&mut bytes)
                    .map_err(|error| file_io_error(&error))?;
                if read == 0 {
                    Ok(None)
                } else {
                    bytes.truncate(read);
                    Ok(Some(LuaString::from(bytes)))
                }
            }
            FileStorage::PipeWrite { .. } => Err(file_stream_error("Bad file descriptor")),
            FileStorage::HostOutput | FileStorage::HostError => {
                Err(file_stream_error("file is not readable"))
            }
            FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
        }
    }

    /// Returns the next byte, or `None` at end of file, and consumes it only
    /// when `consume` accepts it. A rejected byte stays unread, as after C's
    /// `getc` and `ungetc`; standard input peeks into the shared host buffer
    /// instead of seeking, so pipes and terminals work.
    ///
    /// # Errors
    ///
    /// Returns an error for stale, closed, or write-only host files, or an I/O
    /// error.
    fn file_peek_byte(
        &mut self,
        handle: FileHandle,
        consume: impl FnOnce(u8) -> bool,
    ) -> RuntimeResult<Option<u8>> {
        match self.file_mut_open(handle)? {
            FileStorage::Memory { bytes, cursor } => {
                let Some(&byte) = bytes.get(*cursor) else {
                    return Ok(None);
                };
                if consume(byte) {
                    *cursor += 1;
                }
                Ok(Some(byte))
            }
            FileStorage::Named {
                file,
                readable,
                pending,
                ..
            } => {
                if !*readable {
                    return Ok(None);
                }
                flush_pending(file, pending)?;
                let mut byte = [0_u8; 1];
                let read = file
                    .read(&mut byte)
                    .map_err(|error| file_io_error(&error))?;
                if read == 0 {
                    return Ok(None);
                }
                if !consume(byte[0]) {
                    file.seek(SeekFrom::Current(-1))
                        .map_err(|error| file_io_error(&error))?;
                }
                Ok(Some(byte[0]))
            }
            FileStorage::HostInput => {
                let mut input = std::io::stdin().lock();
                let buffer = input.fill_buf().map_err(|error| file_io_error(&error))?;
                let Some(&byte) = buffer.first() else {
                    return Ok(None);
                };
                if consume(byte) {
                    input.consume(1);
                }
                Ok(Some(byte))
            }
            FileStorage::PipeRead { output, .. } => {
                let buffer = output.fill_buf().map_err(|error| file_io_error(&error))?;
                let Some(&byte) = buffer.first() else {
                    return Ok(None);
                };
                if consume(byte) {
                    output.consume(1);
                }
                Ok(Some(byte))
            }
            FileStorage::PipeWrite { .. } => Err(file_stream_error("Bad file descriptor")),
            FileStorage::HostOutput | FileStorage::HostError => {
                Err(file_stream_error("file is not readable"))
            }
            FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
        }
    }

    /// Reads one Lua text line, omitting a trailing line ending.
    ///
    /// `None` means the cursor was already at end of file. An empty line is
    /// returned as `Some("")`.
    ///
    /// # Errors
    ///
    /// Returns an error for stale or closed file handles, or an I/O error for
    /// named files.
    pub fn file_read_line(&mut self, handle: FileHandle) -> RuntimeResult<Option<LuaString>> {
        let file = self.file_mut_open(handle)?;
        let mut line = Vec::new();
        match file {
            FileStorage::Memory { bytes, cursor } => {
                if *cursor >= bytes.len() {
                    return Ok(None);
                }
                while *cursor < bytes.len() {
                    let byte = bytes[*cursor];
                    *cursor += 1;
                    if byte == b'\n' {
                        break;
                    }
                    line.push(byte);
                }
            }
            FileStorage::Named {
                file,
                readable,
                pending,
                ..
            } => {
                if !*readable {
                    return Ok(None);
                }
                flush_pending(file, pending)?;
                let mut byte = [0_u8; 1];
                loop {
                    let read = file
                        .read(&mut byte)
                        .map_err(|error| file_io_error(&error))?;
                    if read == 0 {
                        if line.is_empty() {
                            return Ok(None);
                        }
                        break;
                    }
                    if byte[0] == b'\n' {
                        break;
                    }
                    line.push(byte[0]);
                }
            }
            FileStorage::HostInput => {
                let mut byte = [0_u8; 1];
                loop {
                    let read = std::io::stdin()
                        .read(&mut byte)
                        .map_err(|error| file_io_error(&error))?;
                    if read == 0 {
                        if line.is_empty() {
                            return Ok(None);
                        }
                        break;
                    }
                    if byte[0] == b'\n' {
                        break;
                    }
                    line.push(byte[0]);
                }
            }
            FileStorage::PipeRead { output, .. } => {
                let read = output
                    .read_until(b'\n', &mut line)
                    .map_err(|error| file_io_error(&error))?;
                if read == 0 {
                    return Ok(None);
                }
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
            }
            FileStorage::PipeWrite { .. } => return Err(file_stream_error("Bad file descriptor")),
            FileStorage::HostOutput | FileStorage::HostError => {
                return Err(file_stream_error("file is not readable"));
            }
            FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
        }
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Ok(Some(LuaString::from(line)))
    }

    /// Reads values using Lua 5.1's file formats.
    ///
    /// # Errors
    ///
    /// Returns an error for a closed file, invalid format, or host I/O failure.
    pub fn file_read_values(
        &mut self,
        handle: FileHandle,
        formats: &[Value],
    ) -> RuntimeResult<Vec<Value>> {
        let default = Value::from("*l");
        let formats = if formats.is_empty() {
            std::slice::from_ref(&default)
        } else {
            formats
        };
        let mut values = Vec::with_capacity(formats.len());
        for (index, format) in formats.iter().enumerate() {
            let position = index + 1;
            let value = match format {
                // Lua 5.3 made the `*` prefix optional and only inspects the
                // first option letter.
                Value::String(format) if self.tagged_integers => {
                    let bytes = format.as_bytes();
                    match bytes.strip_prefix(b"*").unwrap_or(bytes).first() {
                        Some(b'n') => self.file_read_numeral(handle)?.unwrap_or(Value::Nil),
                        Some(b'l') => self
                            .file_read_line(handle)?
                            .map_or(Value::Nil, Value::String),
                        Some(b'L') => self
                            .file_read_line_with_newline(handle)?
                            .map_or(Value::Nil, Value::String),
                        Some(b'a') => Value::String(self.file_read_all(handle)?),
                        _ => return Err(RuntimeError::argument_error(position, "invalid format")),
                    }
                }
                Value::Integer(count) if *count >= 0 => {
                    let count = usize::try_from(*count)
                        .map_err(|_| RuntimeError::message("read count is out of range"))?;
                    self.file_read_some(handle, count)?
                        .map_or(Value::Nil, Value::String)
                }
                Value::String(format) if matches!(format.as_bytes(), b"*a" | b"*all") => {
                    Value::String(self.file_read_all(handle)?)
                }
                Value::String(format) if format.as_bytes() == b"*l" => self
                    .file_read_line(handle)?
                    .map_or(Value::Nil, Value::String),
                Value::String(format) if format.as_bytes() == b"*n" => self
                    .file_read_number(handle)?
                    .map_or(Value::Nil, Value::Number),
                Value::Number(count)
                    if count.is_finite()
                        && *count >= 0.0
                        && (count.fract() == 0.0 || !self.tagged_integers) =>
                {
                    // Lua 5.1 and 5.2 truncate a fractional count.
                    let count = count
                        .trunc()
                        .to_string()
                        .parse::<usize>()
                        .map_err(|_| RuntimeError::message("read count is out of range"))?;
                    self.file_read_some(handle, count)?
                        .map_or(Value::Nil, Value::String)
                }
                Value::Number(_) if self.tagged_integers => {
                    return Err(RuntimeError::argument_error(
                        position,
                        "number has no integer representation",
                    ));
                }
                // Lua 5.1 and 5.2 need a `*` before the format letter;
                // LuaJIT reports any unknown format alike.
                Value::String(format)
                    if format.as_bytes().first() == Some(&b'*')
                        || self.number_format == NumberFormat::LuaJit =>
                {
                    return Err(RuntimeError::argument_error(position, "invalid format"));
                }
                Value::String(_) => {
                    return Err(RuntimeError::argument_error(position, "invalid option"));
                }
                value if self.tagged_integers => {
                    return Err(RuntimeError::bad_argument(position, "string", Some(value)));
                }
                _ => return Err(RuntimeError::argument_error(position, "invalid option")),
            };
            let failed = matches!(value, Value::Nil);
            values.push(value);
            if failed {
                break;
            }
        }
        Ok(values)
    }

    /// Reads one line and keeps its terminating newline, as Lua 5.2's `L`
    /// format does.
    fn file_read_line_with_newline(
        &mut self,
        handle: FileHandle,
    ) -> RuntimeResult<Option<LuaString>> {
        let mut line = Vec::new();
        while let Some(byte) = self.file_read_some(handle, 1)? {
            line.extend_from_slice(byte.as_bytes());
            if line.last() == Some(&b'\n') {
                break;
            }
        }
        Ok((!line.is_empty()).then(|| LuaString::from(line)))
    }

    /// Lua 5.3+'s `n` format, ported from liolib's `read_number`: it
    /// consumes the longest prefix that looks like a numeral (at most 200
    /// bytes), leaves the look-ahead byte unread, and converts the text with
    /// Lua's numeral rules, so `0x10` reads as an integer.
    fn file_read_numeral(&mut self, handle: FileHandle) -> RuntimeResult<Option<Value>> {
        let mut scan = NumeralScan::new(handle, NumeralScan::LUA_MAX_LENGTH);
        scan.skip_space(self)?;
        scan.accept(self, b"-+")?;
        let mut digits = 0;
        let mut hexadecimal = false;
        if scan.accept(self, b"0")? {
            if scan.accept(self, b"xX")? {
                hexadecimal = true;
            } else {
                digits = 1;
            }
        }
        digits += scan.digits(self, hexadecimal)?;
        if scan.accept(self, b".")? {
            digits += scan.digits(self, hexadecimal)?;
        }
        if digits > 0 && scan.accept(self, if hexadecimal { b"pP" } else { b"eE" })? {
            scan.accept(self, b"-+")?;
            scan.digits(self, false)?;
        }
        if scan.overflow {
            return Ok(None);
        }
        if let Some(integer) = parse_lua_integer_exact(&scan.text) {
            return Ok(Some(Value::Integer(integer)));
        }
        Ok(parse_lua_number(&scan.text).map(Value::Number))
    }

    /// Lua 5.1's `*n` format, which is C's `fscanf(f, "%lf")`. Ported from
    /// glibc's floating-point conversion: it consumes the longest prefix
    /// that can begin a number (keeping it consumed on failure), leaves the
    /// rejected byte unread, and fails unless the consumed text is complete.
    pub(crate) fn file_read_number(&mut self, handle: FileHandle) -> RuntimeResult<Option<f64>> {
        let mut scan = NumeralScan::new(handle, usize::MAX);
        scan.skip_space(self)?;
        scan.accept(self, b"-+")?;
        let negative = scan.text.first() == Some(&b'-');
        if scan.accept(self, b"nN")? {
            if !(scan.accept(self, b"aA")? && scan.accept(self, b"nN")?) {
                return Ok(None);
            }
            if scan.accept(self, b"(")? {
                while scan.accept_if(self, |byte| byte.is_ascii_alphanumeric() || byte == b'_')? {}
                if !scan.accept(self, b")")? {
                    return Ok(None);
                }
            }
            return Ok(Some(if negative { -f64::NAN } else { f64::NAN }));
        }
        if scan.accept(self, b"iI")? {
            if !(scan.accept(self, b"nN")? && scan.accept(self, b"fF")?) {
                return Ok(None);
            }
            if scan.accept(self, b"iI")? {
                for letters in [b"nN", b"iI", b"tT", b"yY"] {
                    if !scan.accept(self, letters)? {
                        return Ok(None);
                    }
                }
            }
            return Ok(Some(if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }));
        }
        let mut hexadecimal = false;
        let mut got_digit = false;
        if scan.accept(self, b"0")? {
            hexadecimal = scan.accept(self, b"xX")?;
            got_digit = !hexadecimal;
        }
        let exponent: &[u8] = if hexadecimal { b"pP" } else { b"eE" };
        let mut got_dot = false;
        let mut got_exponent = false;
        loop {
            let digit = |byte: u8| {
                byte.is_ascii_digit() || (hexadecimal && !got_exponent && byte.is_ascii_hexdigit())
            };
            if scan.accept_if(self, digit)? {
                got_digit = true;
            } else if got_exponent && scan.text.last().is_some_and(|byte| exponent.contains(byte)) {
                if !scan.accept(self, b"-+")? {
                    break;
                }
            } else if got_digit && !got_exponent && scan.accept(self, exponent)? {
                got_exponent = true;
                got_dot = true;
            } else if !got_dot && scan.accept(self, b".")? {
                got_dot = true;
            } else {
                break;
            }
        }
        if hexadecimal {
            return Ok(parse_lua_number(&scan.text));
        }
        Ok(std::str::from_utf8(&scan.text)
            .ok()
            .and_then(|text| text.parse::<f64>().ok()))
    }

    /// Flushes buffered output for a file.
    ///
    /// # Errors
    ///
    /// Returns an error for a closed file or host I/O failure.
    pub fn file_flush(&mut self, handle: FileHandle) -> RuntimeResult<()> {
        if matches!(&self.file(handle)?.storage, FileStorage::HostOutput) {
            return self.flush_output();
        }
        if matches!(&self.file(handle)?.storage, FileStorage::HostError) {
            return self.flush_error_output();
        }
        match self.file_mut_open(handle)? {
            FileStorage::Named { file, pending, .. } => {
                flush_pending(file, pending)?;
                Ok(())
            }
            FileStorage::PipeWrite { input, .. } => {
                input.flush().map_err(|error| file_io_error(&error))
            }
            FileStorage::PipeRead { .. } => Ok(()),
            FileStorage::Memory { .. }
            | FileStorage::HostInput
            | FileStorage::HostOutput
            | FileStorage::HostError => Ok(()),
            FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
        }
    }

    /// Changes the buffering mode for a named file or host standard stream.
    ///
    /// # Errors
    ///
    /// Returns an error for a closed file, invalid mode or size, or host I/O failure.
    pub fn file_setvbuf(
        &mut self,
        handle: FileHandle,
        mode: &Value,
        size: Option<&Value>,
    ) -> RuntimeResult<()> {
        let Value::String(mode) = mode else {
            return Err(RuntimeError::TypeError {
                operation: "setvbuf",
                expected: "string",
                found: mode.type_name(),
            });
        };
        let size = size.map_or(Ok(8_192_usize), |value| {
            let number = Self::coerce_number(value, "setvbuf")?;
            if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
                return Err(RuntimeError::message("invalid buffer size"));
            }
            number
                .to_string()
                .parse::<usize>()
                .map_err(|_| RuntimeError::message("invalid buffer size"))
        })?;
        let buffering = match mode.as_bytes() {
            b"no" => FileBuffering::No,
            b"full" => FileBuffering::Full(size.max(1)),
            b"line" => FileBuffering::Line(size.max(1)),
            _ => return Err(RuntimeError::message("invalid option")),
        };
        if matches!(&self.file(handle)?.storage, FileStorage::HostOutput) {
            self.output_buffering = buffering;
            return Ok(());
        }
        if matches!(&self.file(handle)?.storage, FileStorage::HostError) {
            self.error_output_buffering = buffering;
            return Ok(());
        }
        match self.file_mut_open(handle)? {
            FileStorage::Named {
                file,
                buffering: current,
                pending,
                ..
            } => {
                flush_pending(file, pending)?;
                *current = buffering;
                Ok(())
            }
            FileStorage::Memory { .. }
            | FileStorage::PipeRead { .. }
            | FileStorage::PipeWrite { .. }
            | FileStorage::HostInput
            | FileStorage::HostOutput
            | FileStorage::HostError => Ok(()),
            FileStorage::Closed => unreachable!("file_mut_open rejects closed files"),
        }
    }

    /// Creates a callable line iterator. Iterators created by `io.lines(name)`
    /// own their file and close it at EOF.
    ///
    /// Profiles whose `lines` accepts read formats store `formats` and pass
    /// them to [`Runtime::file_read_values`] on each call; `function` and
    /// `first_format_argument` name the caller in the too-many-formats error.
    ///
    /// # Errors
    ///
    /// Returns an error for a closed file, too many formats, or arena
    /// allocation failure.
    pub fn create_file_lines_iterator(
        &mut self,
        handle: FileHandle,
        close_at_eof: bool,
        formats: &[Value],
        function: &str,
        first_format_argument: usize,
    ) -> RuntimeResult<Value> {
        let formats = if self.lines_formats == LinesFormats::Ignored {
            &[]
        } else {
            formats
        };
        self.lines_formats
            .check_count(formats.len(), function, first_format_argument)?;
        self.file_open(handle)?;
        let iterator = self.create_table()?;
        let metatable = self.create_table()?;
        let callback = self.register_native(file_line_iterator_next)?;
        self.table_set(iterator, &Value::from("\0file"), Value::File(handle))?;
        self.table_set(
            iterator,
            &Value::from("\0close"),
            Value::Boolean(close_at_eof),
        )?;
        if !formats.is_empty() {
            // Formats live in the iterator table so the collector keeps them
            // alive; the count preserves trailing nil formats.
            for (index, format) in (1_i64..).zip(formats) {
                let key = self.integer_value(index);
                self.table_set(iterator, &key, format.clone())?;
            }
            let count = self.integer_value(i64::try_from(formats.len()).unwrap_or(i64::MAX));
            self.table_set(iterator, &Value::from("\0formats"), count)?;
        }
        self.table_set(iterator, &Value::from("\0done"), Value::Boolean(false))?;
        self.table_set(
            metatable,
            &Value::from("__call"),
            Value::NativeFunction(callback),
        )?;
        self.raw_set_metatable(&Value::Table(iterator), Some(metatable))?;
        Ok(Value::Table(iterator))
    }

    /// Reports whether a file has been closed.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale file handle.
    pub fn file_is_closed(&self, handle: FileHandle) -> RuntimeResult<bool> {
        Ok(matches!(self.file(handle)?.storage, FileStorage::Closed))
    }

    /// Closes a file. Closing an already-closed file is an error.
    ///
    /// # Errors
    ///
    /// Returns an error for stale or already-closed file handles.
    pub fn file_close(&mut self, handle: FileHandle) -> RuntimeResult<()> {
        self.file_close_process(handle).map(|_| ())
    }

    /// Closes a file and returns a process status for `io.popen` handles.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when closing or waiting fails.
    pub fn file_close_process(&mut self, handle: FileHandle) -> RuntimeResult<Option<ExitStatus>> {
        self.file_flush(handle)?;
        let file = self
            .files
            .get_mut(handle.0)
            .ok_or_else(|| RuntimeError::InvalidFile(handle))?;
        if matches!(file.storage, FileStorage::Closed) {
            return Err(RuntimeError::message("attempt to use a closed file"));
        }
        let storage = std::mem::replace(&mut file.storage, FileStorage::Closed);
        let status = match storage {
            FileStorage::PipeRead { mut child, output } => {
                drop(output);
                Some(child.wait().map_err(|error| file_io_error(&error))?)
            }
            FileStorage::PipeWrite { mut child, input } => {
                drop(input);
                Some(child.wait().map_err(|error| file_io_error(&error))?)
            }
            _ => None,
        };
        Ok(status)
    }
}

/// Scanner state for the `n` read formats, mirroring liolib's `RN`. The
/// look-ahead byte is peeked rather than read, so nothing needs unreading.
struct NumeralScan {
    handle: FileHandle,
    text: Vec<u8>,
    max_length: usize,
    overflow: bool,
}

impl NumeralScan {
    /// liolib's `L_MAXLENNUM`.
    const LUA_MAX_LENGTH: usize = 200;

    const fn new(handle: FileHandle, max_length: usize) -> Self {
        Self {
            handle,
            text: Vec::new(),
            max_length,
            overflow: false,
        }
    }

    /// Skips C `isspace` bytes without recording them.
    fn skip_space(&self, runtime: &mut Runtime) -> RuntimeResult<()> {
        let space = |byte: u8| matches!(byte, b' ' | b'\t'..=b'\r');
        while runtime
            .file_peek_byte(self.handle, space)?
            .is_some_and(space)
        {}
        Ok(())
    }

    /// Consumes the look-ahead byte when `accept` matches it (liolib
    /// `test2`). A full buffer refuses the byte and marks the numeral as
    /// overflowed, as liolib's `nextc` does.
    fn accept_if(
        &mut self,
        runtime: &mut Runtime,
        accept: impl Fn(u8) -> bool,
    ) -> RuntimeResult<bool> {
        let full = self.text.len() >= self.max_length;
        let byte = runtime.file_peek_byte(self.handle, |byte| !full && accept(byte))?;
        let Some(byte) = byte.filter(|&byte| accept(byte)) else {
            return Ok(false);
        };
        if full {
            self.overflow = true;
            return Ok(false);
        }
        self.text.push(byte);
        Ok(true)
    }

    fn accept(&mut self, runtime: &mut Runtime, set: &[u8]) -> RuntimeResult<bool> {
        self.accept_if(runtime, |byte| set.contains(&byte))
    }

    fn digits(&mut self, runtime: &mut Runtime, hexadecimal: bool) -> RuntimeResult<usize> {
        let mut count = 0;
        while self.accept_if(runtime, |byte| {
            if hexadecimal {
                byte.is_ascii_hexdigit()
            } else {
                byte.is_ascii_digit()
            }
        })? {
            count += 1;
        }
        Ok(count)
    }
}
