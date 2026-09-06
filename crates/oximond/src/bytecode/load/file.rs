use super::{ChunkLoad, ChunkLoader, LoadInput, LoadPoll};
use crate::bytecode::binary;
use crate::string::LuaString;
use std::io::{self, Read};

/// Loads a small regular file through the compiler's complete-input path.
///
/// At most 8193 bytes are buffered before choosing a path. A file that has
/// grown past the 8192-byte cutoff is replayed into the concurrent driver,
/// including its original prefix. Pipes and stdin should use the streaming
/// driver so a compiler error does not wait for EOF.
///
/// # Errors
/// Returns reader errors. Compiler and bytecode errors are `LoadPoll::Failed`.
pub fn load_small_file_reader(
    loader: &dyn ChunkLoader,
    source_name: &LuaString,
    mut reader: impl Read,
    mode: Option<&[u8]>,
) -> io::Result<LoadPoll> {
    let mut bytes = Vec::with_capacity(8193);
    reader.by_ref().take(8193).read_to_end(&mut bytes)?;
    if bytes.len() > 8192 {
        return load_file_reader_pipelined(
            loader,
            source_name,
            io::Cursor::new(bytes).chain(reader),
            mode,
        );
    }
    let bytes = normalize_complete_file(bytes);
    let is_binary = binary::claims(&bytes);
    if let Some(error) = mode_error(mode, is_binary) {
        return Ok(error);
    }
    Ok(if is_binary {
        decode_binary(&bytes)
    } else {
        loader.load_complete(source_name.clone(), LuaString::from(bytes))
    })
}

fn normalize_complete_file(mut bytes: Vec<u8>) -> Vec<u8> {
    let mut start = usize::from(bytes.starts_with(b"\xef\xbb\xbf")) * 3;
    let shell = bytes.get(start) == Some(&b'#');
    if shell {
        start = bytes[start..]
            .iter()
            .position(|byte| matches!(byte, b'\r' | b'\n'))
            .map_or(bytes.len(), |offset| {
                let ending = start + offset;
                ending
                    + 1
                    + usize::from(bytes[ending] == b'\r' && bytes.get(ending + 1) == Some(&b'\n'))
            });
    }
    let preserve_line = shell && !binary::claims(&bytes[start..]);
    let prefix = usize::from(preserve_line);
    bytes.copy_within(start.., prefix);
    bytes.truncate(bytes.len() - start + prefix);
    if preserve_line {
        bytes[0] = b'\n';
    }
    bytes
}

fn mode_error(mode: Option<&[u8]>, is_binary: bool) -> Option<LoadPoll> {
    mode.filter(|mode| !mode.contains(&if is_binary { b'b' } else { b't' }))
        .map(|_| {
            LoadPoll::Failed(LuaString::from(if is_binary {
                "attempt to load a binary chunk (mode is 't')"
            } else {
                "attempt to load a text chunk (mode is 'b')"
            }))
        })
}

/// Loads a host file with bounded reads and synchronous compiler delivery.
///
/// A UTF-8 BOM and an initial shell comment are removed. Text retains a blank
/// first line for the shell comment. Each read reaches the compiler before the
/// next read, except while identifying the short prefix or skipping the comment.
/// Binary bytecode is the exception: the binary decoder needs the complete file,
/// so recognized Oximond bytecode is buffered until EOF.
///
/// # Errors
/// Returns reader errors. Compiler and bytecode errors are `LoadPoll::Failed`.
pub fn load_file_reader(
    loader: &dyn ChunkLoader,
    source_name: &LuaString,
    reader: impl Read,
    mode: Option<&[u8]>,
) -> io::Result<LoadPoll> {
    drive_file_reader(loader, source_name, reader, mode, false)
}

/// Loads a read-ahead-safe file with a concurrent parser/compiler when the
/// loader supports it. Prefix handling, modes, binary decoding and errors
/// match [`load_file_reader`]. Loaders without a pipeline remain synchronous.
///
/// # Errors
/// Returns reader errors. Compiler and bytecode errors are `LoadPoll::Failed`.
pub fn load_file_reader_pipelined(
    loader: &dyn ChunkLoader,
    source_name: &LuaString,
    reader: impl Read,
    mode: Option<&[u8]>,
) -> io::Result<LoadPoll> {
    drive_file_reader(loader, source_name, reader, mode, true)
}

#[allow(clippy::too_many_lines)] // One transition loop handles the file prefix and read scheduling.
#[allow(clippy::missing_panics_doc)] // The assertion checks an internal state callers cannot change.
fn drive_file_reader(
    loader: &dyn ChunkLoader,
    source_name: &LuaString,
    mut reader: impl Read,
    mode: Option<&[u8]>,
    pipelined: bool,
) -> io::Result<LoadPoll> {
    const MAGIC: &[u8] = b"\x1bOXIMOND";
    const BOM: &[u8] = b"\xef\xbb\xbf";
    enum Prefix {
        Bom,
        Hash,
        Comment,
        AfterCr,
        Magic,
        Done,
    }
    let mut prefix = Prefix::Bom;
    let mut pending = Vec::with_capacity(MAGIC.len());
    let mut shell = false;
    let mut binary_bytes = None::<Vec<u8>>;
    let mut session: Option<Box<dyn ChunkLoad>> = None;
    let mut buffer = [0; 8192];
    loop {
        let count = loop {
            if let Some(result) = session.as_mut().and_then(|active| active.poll()) {
                return Ok(result.into_poll());
            }
            match reader.read(&mut buffer) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => {
                    if let Some(result) = session.as_mut().and_then(|active| active.poll()) {
                        return Ok(result.into_poll());
                    }
                    // Dropping the session cancels workers. A failed read is
                    // never presented as EOF and cannot publish partial code.
                    return Err(error);
                }
                Ok(count) => break count,
            }
        };
        let eof = count == 0;
        if matches!(prefix, Prefix::Done) {
            if let Some(bytes) = &mut binary_bytes {
                bytes.extend_from_slice(&buffer[..count]);
                if eof {
                    return Ok(decode_binary(bytes));
                }
            } else {
                let active = session.take().expect("text session started");
                if eof {
                    return Ok(active.resume(LoadInput::End));
                }
                session = Some(
                    match active.resume(LoadInput::Bytes(LuaString::from(&buffer[..count]))) {
                        LoadPoll::NeedInput(next) => next,
                        terminal => return Ok(terminal),
                    },
                );
            }
            continue;
        }
        let mut output = Vec::with_capacity(count + MAGIC.len() + 1);
        for (index, &byte) in buffer[..count].iter().enumerate() {
            let mut current = Some(byte);
            while let Some(byte) = current.take() {
                match prefix {
                    Prefix::Bom => {
                        pending.push(byte);
                        if !BOM.starts_with(&pending) {
                            // The first byte also decides whether this is a shell comment.
                            prefix = if pending[0] == b'#' {
                                shell = true;
                                pending.clear();
                                Prefix::Comment
                            } else {
                                Prefix::Magic
                            };
                        } else if pending.len() == BOM.len() {
                            pending.clear();
                            prefix = Prefix::Hash;
                        }
                    }
                    Prefix::Hash => {
                        if byte == b'#' {
                            shell = true;
                            prefix = Prefix::Comment;
                        } else {
                            prefix = Prefix::Magic;
                            current = Some(byte);
                        }
                    }
                    Prefix::Comment => {
                        if byte == b'\n' {
                            prefix = Prefix::Magic;
                        } else if byte == b'\r' {
                            prefix = Prefix::AfterCr;
                        }
                    }
                    Prefix::AfterCr => {
                        prefix = Prefix::Magic;
                        if byte != b'\n' {
                            current = Some(byte);
                        }
                    }
                    Prefix::Magic => pending.push(byte),
                    Prefix::Done => output.push(byte),
                }
                if matches!(prefix, Prefix::Magic)
                    && (!MAGIC.starts_with(&pending) || pending.len() == MAGIC.len())
                {
                    let is_binary = binary::claims(&pending);
                    if let Some(error) = mode_error(mode, is_binary) {
                        return Ok(error);
                    }
                    if is_binary {
                        binary_bytes = Some(Vec::new());
                    } else {
                        session = Some(begin_text(loader, source_name, pipelined));
                        if shell {
                            output.push(b'\n');
                        }
                    }
                    output.append(&mut pending);
                    prefix = Prefix::Done;
                }
            }
            if matches!(prefix, Prefix::Done) {
                output.extend_from_slice(&buffer[index + 1..count]);
                break;
            }
        }
        if eof && !matches!(prefix, Prefix::Done) {
            if let Some(error) = mode_error(mode, false) {
                return Ok(error);
            }
            session = Some(begin_text(loader, source_name, pipelined));
            if shell {
                output.push(b'\n');
            }
            output.append(&mut pending);
        }
        if let Some(bytes) = &mut binary_bytes {
            bytes.extend_from_slice(&output);
            if eof {
                return Ok(decode_binary(bytes));
            }
        } else if let Some(mut active) = session.take() {
            if !output.is_empty() {
                active = match active.resume(LoadInput::Bytes(LuaString::from(output))) {
                    LoadPoll::NeedInput(next) => next,
                    terminal => return Ok(terminal),
                };
            }
            if eof {
                return Ok(active.resume(LoadInput::End));
            }
            session = Some(active);
        }
    }
}

fn begin_text(
    loader: &dyn ChunkLoader,
    source_name: &LuaString,
    pipelined: bool,
) -> Box<dyn ChunkLoad> {
    if pipelined && let Some(session) = loader.begin_pipelined(source_name.clone()) {
        return session;
    }
    loader.begin(source_name.clone())
}

fn decode_binary(bytes: &[u8]) -> LoadPoll {
    match binary::load(bytes) {
        Ok(chunk) => LoadPoll::Loaded(chunk),
        Err(error) => LoadPoll::Failed(LuaString::from(error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    type Events = Rc<RefCell<Vec<String>>>;
    struct Loader {
        events: Events,
        fail_early: bool,
    }
    impl ChunkLoader for Loader {
        fn begin(&self, _: LuaString) -> Box<dyn super::super::ChunkLoad> {
            self.events.borrow_mut().push("begin".into());
            Box::new(Session {
                events: self.events.clone(),
                bytes: Vec::new(),
                fail_early: self.fail_early,
            })
        }
        fn load_complete(&self, _: LuaString, _: LuaString) -> LoadPoll {
            panic!("file text must stream")
        }
    }
    struct Session {
        events: Events,
        bytes: Vec<u8>,
        fail_early: bool,
    }
    impl super::super::ChunkLoad for Session {
        fn resume(mut self: Box<Self>, input: LoadInput) -> LoadPoll {
            match input {
                LoadInput::Bytes(bytes) => {
                    self.events.borrow_mut().push("bytes".into());
                    self.bytes.extend_from_slice(bytes.as_bytes());
                    if self.fail_early {
                        LoadPoll::Failed(LuaString::from("syntax error"))
                    } else {
                        LoadPoll::NeedInput(self)
                    }
                }
                LoadInput::End => {
                    self.events.borrow_mut().push("end".into());
                    LoadPoll::Failed(LuaString::from(self.bytes))
                }
            }
        }
    }
    struct Reader {
        events: Events,
        reads: VecDeque<io::Result<Vec<u8>>>,
    }
    impl Read for Reader {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            self.events.borrow_mut().push("read".into());
            let bytes = self.reads.pop_front().unwrap_or_else(|| Ok(Vec::new()))?;
            assert!(bytes.len() <= output.len());
            output[..bytes.len()].copy_from_slice(&bytes);
            Ok(bytes.len())
        }
    }
    fn message(poll: LoadPoll) -> Vec<u8> {
        match poll {
            LoadPoll::Failed(message) => message.as_bytes().to_vec(),
            _ => panic!("expected failure"),
        }
    }
    fn run(reads: Vec<io::Result<Vec<u8>>>, mode: Option<&[u8]>) -> (Vec<u8>, Events) {
        let events = Events::default();
        let loader = Loader {
            events: events.clone(),
            fail_early: false,
        };
        let reader = Reader {
            events: events.clone(),
            reads: reads.into(),
        };
        let result = load_file_reader(&loader, &LuaString::from("file"), reader, mode).unwrap();
        (message(result), events)
    }
    #[test]
    fn delivers_each_read_before_reading_again() {
        let (bytes, events) = run(vec![Ok(b"return ".to_vec()), Ok(b"1".to_vec())], None);
        assert_eq!(bytes, b"return 1");
        assert_eq!(
            *events.borrow(),
            ["read", "begin", "bytes", "read", "bytes", "read", "end"]
        );
    }
    #[test]
    fn syntax_failure_stops_before_next_read_or_error() {
        let events = Events::default();
        let loader = Loader {
            events: events.clone(),
            fail_early: true,
        };
        let reader = Reader {
            events: events.clone(),
            reads: vec![Ok(b"bad".to_vec()), Err(io::Error::other("must not read"))].into(),
        };
        assert_eq!(
            message(load_file_reader(&loader, &LuaString::from("file"), reader, None).unwrap()),
            b"syntax error"
        );
        assert_eq!(*events.borrow(), ["read", "begin", "bytes"]);
    }
    #[test]
    fn retries_interrupted_and_propagates_read_errors() {
        let (bytes, _) = run(
            vec![
                Err(io::Error::from(io::ErrorKind::Interrupted)),
                Ok(b"x".to_vec()),
            ],
            None,
        );
        assert_eq!(bytes, b"x");
        let events = Events::default();
        let loader = Loader {
            events: events.clone(),
            fail_early: false,
        };
        let reader = Reader {
            events,
            reads: vec![Ok(b"x".to_vec()), Err(io::Error::other("broken"))].into(),
        };
        let result = load_file_reader(&loader, &LuaString::from("file"), reader, None);
        assert!(matches!(result, Err(error) if error.to_string() == "broken"));
    }
    #[test]
    fn strips_prefixes_at_every_split_and_with_single_byte_reads() {
        for (input, expected) in [
            (b"\xef\xbb\xbfreturn 1".as_slice(), b"return 1".as_slice()),
            (b"#shell\nreturn 1", b"\nreturn 1"),
            (b"\xef\xbb\xbf#shell\r\nreturn 1", b"\nreturn 1"),
            (b"#shell\rreturn 1", b"\nreturn 1"),
            (b"#shell", b"\n"),
            (b"\xef\xbb", b"\xef\xbb"),
            (b"\x1bOX", b"\x1bOX"),
            (b"", b""),
        ] {
            for split in 1..input.len() {
                let (bytes, _) = run(
                    vec![Ok(input[..split].to_vec()), Ok(input[split..].to_vec())],
                    None,
                );
                assert_eq!(bytes, expected, "split {split} of {input:?}");
            }
            let (bytes, _) = run(input.iter().map(|byte| Ok(vec![*byte])).collect(), None);
            assert_eq!(bytes, expected);
        }
    }
    #[test]
    fn loads_complete_binary_without_starting_text_compiler() {
        let chunk =
            crate::bytecode::Chunk::new(1, Vec::new(), vec![crate::bytecode::Instruction::halt()]);
        let bytes = binary::dump(&chunk).unwrap();
        let events = Events::default();
        let loader = Loader {
            events: events.clone(),
            fail_early: false,
        };
        let mut reads = vec![Ok(b"#shell\r\n".to_vec())];
        reads.extend(bytes.iter().map(|byte| Ok(vec![*byte])));
        let reader = Reader {
            events: events.clone(),
            reads: reads.into(),
        };
        match load_file_reader(&loader, &LuaString::from("file"), reader, None).unwrap() {
            LoadPoll::Loaded(decoded) => assert_eq!(decoded, chunk),
            _ => panic!("expected binary chunk"),
        }
        assert!(events.borrow().iter().all(|event| event == "read"));
    }

    #[test]
    fn discards_long_shell_comment_in_bounded_reads() {
        let mut reads = vec![Ok(b"#".to_vec())];
        reads.extend((0..32).map(|_| Ok(vec![b'x'; 8192])));
        reads.push(Ok(b"\r\nreturn 1".to_vec()));
        let (bytes, events) = run(reads, None);
        assert_eq!(bytes, b"\nreturn 1");
        assert_eq!(
            events
                .borrow()
                .iter()
                .filter(|event| *event == "bytes")
                .count(),
            1
        );
    }

    #[test]
    fn binary_prefix_and_modes() {
        let raw = b"\x1bOXIMOND";
        for prefix in [
            b"".as_slice(),
            b"\xef\xbb\xbf",
            b"#shell\n",
            b"\xef\xbb\xbf#shell\r\n",
        ] {
            let mut input = prefix.to_vec();
            input.extend_from_slice(raw);
            for split in 1..input.len() {
                let reads = || vec![Ok(input[..split].to_vec()), Ok(input[split..].to_vec())];
                let (bytes, events) = run(reads(), Some(b"b"));
                assert_eq!(bytes, binary::load(raw).unwrap_err().to_string().as_bytes());
                assert!(!events.borrow().iter().any(|event| event == "begin"));
                let (bytes, _) = run(reads(), Some(b"t"));
                assert_eq!(bytes, b"attempt to load a binary chunk (mode is 't')");
            }
        }
        for mode in [b"b".as_slice(), b""] {
            let (bytes, _) = run(vec![Ok(b"return 1".to_vec())], Some(mode));
            assert_eq!(bytes, b"attempt to load a text chunk (mode is 'b')");
        }
    }
    struct PipelineLoader {
        events: Events,
        ready: Rc<RefCell<bool>>,
        fail_on_enqueue: bool,
    }
    struct PipelineSession {
        events: Events,
        ready: Rc<RefCell<bool>>,
        fail_on_enqueue: bool,
    }
    impl ChunkLoader for PipelineLoader {
        fn begin(&self, _: LuaString) -> Box<dyn ChunkLoad> {
            panic!("pipeline-supported loader should use its factory");
        }
        fn begin_pipelined(&self, _: LuaString) -> Option<Box<dyn ChunkLoad>> {
            self.events.borrow_mut().push("pipeline".into());
            Some(Box::new(PipelineSession {
                events: self.events.clone(),
                ready: self.ready.clone(),
                fail_on_enqueue: self.fail_on_enqueue,
            }))
        }
    }
    impl ChunkLoad for PipelineSession {
        fn poll(&mut self) -> Option<super::super::LoadResult> {
            if *self.ready.borrow() {
                self.events.borrow_mut().push("terminal".into());
                Some(super::super::LoadResult::Failed(LuaString::from(
                    "worker syntax error",
                )))
            } else {
                None
            }
        }
        fn resume(self: Box<Self>, input: LoadInput) -> LoadPoll {
            match input {
                LoadInput::Bytes(_) => {
                    self.events.borrow_mut().push("enqueue".into());
                    *self.ready.borrow_mut() = self.fail_on_enqueue;
                    LoadPoll::NeedInput(self)
                }
                LoadInput::End => panic!("terminal worker should be polled before EOF"),
            }
        }
    }
    impl Drop for PipelineSession {
        fn drop(&mut self) {
            self.events.borrow_mut().push("drop".into());
        }
    }

    #[test]
    fn pipeline_polls_worker_failure_before_another_read() {
        let events = Events::default();
        let loader = PipelineLoader {
            events: events.clone(),
            ready: Rc::default(),
            fail_on_enqueue: true,
        };
        let reader = Reader {
            events: events.clone(),
            reads: vec![Ok(b"bad".to_vec()), Err(io::Error::other("must not read"))].into(),
        };
        assert_eq!(
            message(
                load_file_reader_pipelined(&loader, &LuaString::from("file"), reader, None)
                    .unwrap()
            ),
            b"worker syntax error"
        );
        assert_eq!(
            *events.borrow(),
            ["read", "pipeline", "enqueue", "terminal", "drop"]
        );
    }

    #[test]
    fn pipeline_wrapper_keeps_unsupported_loaders_synchronous() {
        let events = Events::default();
        let loader = Loader {
            events: events.clone(),
            fail_early: true,
        };
        let reader = Reader {
            events: events.clone(),
            reads: vec![Ok(b"bad".to_vec()), Err(io::Error::other("must not read"))].into(),
        };
        assert_eq!(
            message(
                load_file_reader_pipelined(&loader, &LuaString::from("file"), reader, None)
                    .unwrap()
            ),
            b"syntax error"
        );
        assert_eq!(*events.borrow(), ["read", "begin", "bytes"]);
    }
    #[test]
    fn pipeline_read_failure_cancels_without_synthetic_eof() {
        let events = Events::default();
        let loader = PipelineLoader {
            events: events.clone(),
            ready: Rc::default(),
            fail_on_enqueue: false,
        };
        let reader = Reader {
            events: events.clone(),
            reads: vec![
                Ok(b"return 1".to_vec()),
                Err(io::Error::other("read failed")),
            ]
            .into(),
        };
        let result = load_file_reader_pipelined(&loader, &LuaString::from("file"), reader, None);
        assert!(matches!(result, Err(error) if error.to_string() == "read failed"));
        assert_eq!(
            *events.borrow(),
            ["read", "pipeline", "enqueue", "read", "drop"]
        );
    }

    #[test]
    fn pipeline_worker_failure_observed_during_read_precedes_read_error() {
        struct FailingRead {
            reads: usize,
            ready: Rc<RefCell<bool>>,
        }
        impl Read for FailingRead {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                self.reads += 1;
                if self.reads == 1 {
                    output[..3].copy_from_slice(b"bad");
                    Ok(3)
                } else {
                    *self.ready.borrow_mut() = true;
                    Err(io::Error::other("read failed"))
                }
            }
        }
        let events = Events::default();
        let ready = Rc::default();
        let loader = PipelineLoader {
            events: events.clone(),
            ready: Rc::clone(&ready),
            fail_on_enqueue: false,
        };
        let reader = FailingRead { reads: 0, ready };
        assert_eq!(
            message(
                load_file_reader_pipelined(&loader, &LuaString::from("file"), reader, None)
                    .unwrap()
            ),
            b"worker syntax error"
        );
        assert_eq!(
            *events.borrow(),
            ["pipeline", "enqueue", "terminal", "drop"]
        );
    }
    struct CompleteLoader {
        events: Events,
    }
    impl ChunkLoader for CompleteLoader {
        fn begin(&self, _: LuaString) -> Box<dyn ChunkLoad> {
            panic!("small files should compile complete input");
        }
        fn load_complete(&self, _: LuaString, source: LuaString) -> LoadPoll {
            self.events.borrow_mut().push("complete".into());
            LoadPoll::Failed(source)
        }
    }

    #[test]
    fn small_file_complete_normalization_matches_streaming() {
        for input in [
            b"return 1".as_slice(),
            b"",
            b"#shell",
            b"#shell\nreturn 1",
            b"#shell\rreturn 1",
            b"#shell\r\nreturn 1",
            b"\xef\xbb\xbfreturn 1",
            b"\xef\xbb\xbf#shell\r\nreturn 1",
            b"\xef\xbb",
            b"\x1bOX",
        ] {
            let events = Events::default();
            let loader = CompleteLoader {
                events: events.clone(),
            };
            let actual = message(
                load_small_file_reader(
                    &loader,
                    &LuaString::from("file"),
                    io::Cursor::new(input),
                    None,
                )
                .unwrap(),
            );
            let (expected, _) = run(vec![Ok(input.to_vec())], None);
            assert_eq!(actual, expected, "{input:?}");
            assert_eq!(*events.borrow(), ["complete"]);
        }
        let chunk =
            crate::bytecode::Chunk::new(1, Vec::new(), vec![crate::bytecode::Instruction::halt()]);
        let bytes = binary::dump(&chunk).unwrap();
        for prefix in [
            b"".as_slice(),
            b"#shell\r",
            b"#shell\r\n",
            b"\xef\xbb\xbf#shell\n",
        ] {
            let source = [prefix, &bytes].concat();
            let events = Events::default();
            let loader = CompleteLoader {
                events: events.clone(),
            };
            match load_small_file_reader(
                &loader,
                &LuaString::from("file"),
                io::Cursor::new(&source),
                Some(b"b"),
            )
            .unwrap()
            {
                LoadPoll::Loaded(decoded) => assert_eq!(decoded, chunk),
                _ => panic!("binary file"),
            }
            assert!(events.borrow().is_empty());
            assert_eq!(
                message(
                    load_small_file_reader(
                        &loader,
                        &LuaString::from("file"),
                        io::Cursor::new(source),
                        Some(b"t")
                    )
                    .unwrap()
                ),
                b"attempt to load a binary chunk (mode is 't')"
            );
        }
    }

    #[test]
    fn small_file_default_complete_method_matches_streaming() {
        struct DefaultLoader {
            events: Events,
        }
        impl ChunkLoader for DefaultLoader {
            fn begin(&self, _: LuaString) -> Box<dyn ChunkLoad> {
                Box::new(Session {
                    events: self.events.clone(),
                    bytes: Vec::new(),
                    fail_early: false,
                })
            }
        }
        let events = Events::default();
        let loader = DefaultLoader {
            events: events.clone(),
        };
        assert_eq!(
            message(
                load_small_file_reader(
                    &loader,
                    &LuaString::from("file"),
                    io::Cursor::new(b"#shell\r\nreturn 1"),
                    None
                )
                .unwrap()
            ),
            b"\nreturn 1"
        );
        assert_eq!(*events.borrow(), ["bytes", "end"]);
    }

    #[test]
    fn growing_small_file_replays_prefix_after_bounded_probe() {
        use std::cell::Cell;
        struct GrowingReader {
            inner: io::Cursor<Vec<u8>>,
            count: Rc<Cell<usize>>,
        }
        impl Read for GrowingReader {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                let count = self.inner.read(output)?;
                self.count.set(self.count.get() + count);
                Ok(count)
            }
        }
        struct GrowingLoader {
            count: Rc<Cell<usize>>,
            events: Events,
        }
        impl ChunkLoader for GrowingLoader {
            fn begin(&self, _: LuaString) -> Box<dyn ChunkLoad> {
                panic!("growth selects pipeline")
            }
            fn begin_pipelined(&self, _: LuaString) -> Option<Box<dyn ChunkLoad>> {
                assert_eq!(self.count.get(), 8193, "probe must stay bounded");
                self.events.borrow_mut().push("pipeline".into());
                Some(Box::new(Session {
                    events: self.events.clone(),
                    bytes: Vec::new(),
                    fail_early: false,
                }))
            }
            fn load_complete(&self, _: LuaString, _: LuaString) -> LoadPoll {
                panic!("growth cannot use complete path")
            }
        }
        let body = vec![b'x'; 10000];
        let source = [b"\xef\xbb\xbf#shell\r\n".as_slice(), &body].concat();
        let count = Rc::new(Cell::new(0));
        let events = Events::default();
        let loader = GrowingLoader {
            count: count.clone(),
            events: events.clone(),
        };
        let reader = GrowingReader {
            inner: io::Cursor::new(source.clone()),
            count: count.clone(),
        };
        assert_eq!(
            message(
                load_small_file_reader(&loader, &LuaString::from("file"), reader, None).unwrap()
            ),
            [b"\n".as_slice(), &body].concat()
        );
        assert_eq!(count.get(), source.len());
        assert_eq!(
            events.borrow().first().map(String::as_str),
            Some("pipeline")
        );
    }
}
