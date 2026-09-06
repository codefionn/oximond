//! Bounded file-loading pipeline. Reader, parser, and compiler run separately.

use super::{
    CALLER_STACK_DEPTH, COMPILE_THREAD_BASE_STACK_BYTES, CompilerConfig, IncrementalCompilation,
    STACK_BYTES_PER_LEVEL,
};
use crate::bytecode::load::{ChunkLoad, LoadInput, LoadPoll, LoadResult};
use crate::string::LuaString;
use crate::syntax::{self as syntax, Span, Statement};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};

const QUEUE_CAPACITY: usize = 2;
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

pub(super) fn begin(source_name: LuaString, config: CompilerConfig) -> Option<Box<dyn ChunkLoad>> {
    start(source_name, config, Hooks::default()).map(|proxy| Box::new(proxy) as Box<dyn ChunkLoad>)
}

struct Proxy {
    input: Option<SyncSender<LoadInput>>,
    cancelled: Arc<AtomicBool>,
    result: Receiver<LoadResult>,
    parser: Option<JoinHandle<()>>,
    compiler: Option<JoinHandle<()>>,
}

impl Proxy {
    fn receive(&self) -> LoadResult {
        self.result.recv().unwrap_or_else(|_| worker_failure())
    }
}

impl ChunkLoad for Proxy {
    fn poll(&mut self) -> Option<LoadResult> {
        match self.result.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(worker_failure()),
        }
    }

    fn resume(mut self: Box<Self>, input: LoadInput) -> LoadPoll {
        if let Some(result) = self.poll() {
            return result.into_poll();
        }
        let end = matches!(input, LoadInput::End);
        if self
            .input
            .as_ref()
            .expect("live proxy has input")
            .send(input)
            .is_err()
        {
            return self.receive().into_poll();
        }
        if end {
            self.input.take();
            self.receive().into_poll()
        } else if let Some(result) = self.poll() {
            result.into_poll()
        } else {
            LoadPoll::NeedInput(self)
        }
    }
}

impl Drop for Proxy {
    fn drop(&mut self) {
        // Cancellation stops both workers at their next batch boundary.
        // Closing input wakes the parser and closes its downstream channel.
        self.cancelled.store(true, Ordering::Release);
        self.input.take();
        if let Some(parser) = self.parser.take() {
            let _ = parser.join();
        }
        if let Some(compiler) = self.compiler.take() {
            let _ = compiler.join();
        }
    }
}

fn worker_failure() -> LoadResult {
    LoadResult::Failed(LuaString::from("source loading worker panicked"))
}

enum ParsedInput {
    Batch {
        source: LuaString,
        statements: Vec<Statement>,
        line_start_from: usize,
        line_starts: Vec<usize>,
    },
    End(syntax::Chunk),
    Failed {
        source: LuaString,
        error: syntax::SyntaxError,
    },
    Panicked,
}

fn start(source_name: LuaString, config: CompilerConfig, hooks: Hooks) -> Option<Proxy> {
    let (raw_sender, raw_receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
    let (parsed_sender, parsed_receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
    let (result_sender, result_receiver) = mpsc::sync_channel(1);
    let compiler_hooks = hooks.clone();
    let cancelled = Arc::new(AtomicBool::new(false));
    let compiler_cancelled = cancelled.clone();
    let parser_cancelled = cancelled.clone();
    let compiler = thread::Builder::new()
        .name("oximond-file-compile".into())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(move || {
            let _exit = ExitHook::new(compiler_hooks.clone(), Stage::CompilerExit);
            let result = catch_unwind(AssertUnwindSafe(|| {
                compiler_worker(
                    &source_name,
                    config,
                    parsed_receiver,
                    &compiler_hooks,
                    &compiler_cancelled,
                )
            }));
            if let Ok(Some(result)) = result {
                let _ = result_sender.send(result);
            } else if result.is_err() {
                let _ = result_sender.send(worker_failure());
            }
        })
        .ok()?;
    let parser = thread::Builder::new()
        .name("oximond-file-parse".into())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(move || {
            let _exit = ExitHook::new(hooks.clone(), Stage::ParserExit);
            if catch_unwind(AssertUnwindSafe(|| {
                parser_worker(
                    config,
                    raw_receiver,
                    &parsed_sender,
                    &hooks,
                    &parser_cancelled,
                );
            }))
            .is_err()
            {
                let _ = parsed_sender.send(ParsedInput::Panicked);
            }
        });
    if let Ok(parser) = parser {
        Some(Proxy {
            input: Some(raw_sender),
            cancelled,
            result: result_receiver,
            parser: Some(parser),
            compiler: Some(compiler),
        })
    } else {
        // A failed spawn drops its closure and parsed sender, waking the
        // compiler. Complete cleanup before falling back to the caller.
        drop(raw_sender);
        let _ = compiler.join();
        None
    }
}

#[allow(clippy::needless_pass_by_value)] // Owning the receiver closes upstream on return or unwind.
fn parser_worker(
    config: CompilerConfig,
    input: Receiver<LoadInput>,
    output: &SyncSender<ParsedInput>,
    hooks: &Hooks,
    cancelled: &AtomicBool,
) {
    let mut parser = syntax::StreamingParser::with_profile(config.compatibility_profile());
    let mut known_lines = 0_usize;
    while let Ok(input) = input.recv() {
        if cancelled.load(Ordering::Acquire) {
            return;
        }
        match input {
            LoadInput::Bytes(source) => {
                hooks.call(Stage::Parsing);
                let poll = parser.push(source.as_bytes());
                let statements = match poll {
                    syntax::SourceParsePoll::NeedInput => {
                        match parser.take_completed_statements() {
                            Ok(statements) => statements,
                            Err(error) => {
                                let _ = output.send(ParsedInput::Failed { source, error });
                                return;
                            }
                        }
                    }
                    syntax::SourceParsePoll::Failed(error) => {
                        let _ = output.send(ParsedInput::Failed { source, error });
                        return;
                    }
                    syntax::SourceParsePoll::Parsed(_) => {
                        unreachable!("bytes cannot end file parsing")
                    }
                };
                let line_start_from = known_lines.saturating_sub(1);
                let line_starts = parser.line_starts()[line_start_from..].to_vec();
                known_lines = parser.line_starts().len();
                hooks.call(Stage::Parsed);
                if output
                    .send(ParsedInput::Batch {
                        source,
                        statements,
                        line_start_from,
                        line_starts,
                    })
                    .is_err()
                {
                    return;
                }
            }
            LoadInput::End => {
                let terminal = match parser.end_owned() {
                    Ok(chunk) => ParsedInput::End(chunk),
                    Err(error) => ParsedInput::Failed {
                        source: LuaString::from(""),
                        error,
                    },
                };
                let _ = output.send(terminal);
                return;
            }
        }
    }
}

#[allow(clippy::needless_pass_by_value)] // Owning the receiver unblocks a queued parser send during unwind.
fn compiler_worker(
    source_name: &LuaString,
    config: CompilerConfig,
    input: Receiver<ParsedInput>,
    hooks: &Hooks,
    cancelled: &AtomicBool,
) -> Option<LoadResult> {
    let mut compilation = IncrementalCompilation::default();
    let mut source = Vec::new();
    let mut tree = RetainedTree(Some(syntax::Chunk {
        statements: Vec::new(),
        span: Span { start: 0, end: 0 },
        line_starts: Vec::new(),
    }));
    while let Ok(input) = input.recv() {
        if cancelled.load(Ordering::Acquire) {
            return None;
        }
        match input {
            ParsedInput::Batch {
                source: bytes,
                statements,
                line_start_from,
                line_starts,
            } => {
                source.extend_from_slice(bytes.as_bytes());
                let tree = tree.0.as_mut().expect("worker retains tree");
                tree.statements.extend(statements);
                tree.line_starts.truncate(line_start_from);
                tree.line_starts.extend(line_starts);
                hooks.call(Stage::Compiling);
                compilation.advance(
                    source_name,
                    config,
                    &tree.statements,
                    &tree.line_starts,
                    false,
                );
                hooks.call(Stage::Compiled);
            }
            ParsedInput::End(final_tree) => {
                let mut tree = tree.0.take().expect("worker retains tree");
                tree.statements.extend(final_tree.statements);
                tree.span = final_tree.span;
                tree.line_starts = final_tree.line_starts;
                return Some(
                    match compilation.finish(source_name.clone(), tree, config) {
                        Ok(chunk) => LoadResult::Loaded(chunk),
                        Err(error) => LoadResult::Failed(error.diagnostic(source_name, &source)),
                    },
                );
            }
            ParsedInput::Failed {
                source: bytes,
                error,
            } => {
                source.extend_from_slice(bytes.as_bytes());
                return Some(LoadResult::Failed(
                    error.with_source(source_name, &source).diagnostic(),
                ));
            }
            ParsedInput::Panicked => return Some(worker_failure()),
        }
    }
    None
}

/// Cancellation and unwinding still need to drop deeply chained source trees
/// on a stack sized for them, just as the complete-input compiler does.
struct RetainedTree(Option<syntax::Chunk>);
impl Drop for RetainedTree {
    fn drop(&mut self) {
        let Some(mut tree) = self.0.take() else {
            return;
        };
        let depth = tree.nesting_depth();
        if depth <= CALLER_STACK_DEPTH {
            return;
        }
        let stack_bytes = depth
            .saturating_mul(STACK_BYTES_PER_LEVEL)
            .saturating_add(COMPILE_THREAD_BASE_STACK_BYTES);
        let _ = thread::scope(|scope| {
            thread::Builder::new()
                .name("oximond-tree-drop".into())
                .stack_size(stack_bytes)
                .spawn_scoped(scope, || drop(core::mem::take(&mut tree.statements)))
                .map(thread::ScopedJoinHandle::join)
        });
    }
}

#[derive(Clone, Default)]
struct Hooks {
    #[cfg(test)]
    callback: Option<std::sync::Arc<dyn Fn(Stage) + Send + Sync>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    Parsing,
    Parsed,
    Compiling,
    Compiled,
    ParserExit,
    CompilerExit,
}
impl Hooks {
    #[allow(clippy::unused_self)] // Non-test hooks are zero-sized; keep the same method in both builds.
    fn call(&self, stage: Stage) {
        #[cfg(test)]
        if let Some(callback) = &self.callback {
            callback(stage);
        }
        #[cfg(not(test))]
        let _ = stage;
    }
}
struct ExitHook {
    hooks: Hooks,
    stage: Stage,
}
impl ExitHook {
    fn new(hooks: Hooks, stage: Stage) -> Self {
        Self { hooks, stage }
    }
}
impl Drop for ExitHook {
    fn drop(&mut self) {
        self.hooks.call(self.stage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::load::ChunkLoader;
    use crate::compat::CompatibilityProfile;
    use crate::compiler::{CompilerLoader, compile_named};
    use std::sync::{Mutex, atomic::AtomicUsize};
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(10);

    // Releases a blocked worker even when an assertion unwinds the test.
    struct Release(Option<mpsc::Sender<()>>);
    impl Drop for Release {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    fn feed(proxy: Box<dyn ChunkLoad>, source: &[u8]) -> Box<dyn ChunkLoad> {
        match proxy.resume(LoadInput::Bytes(LuaString::from(source))) {
            LoadPoll::NeedInput(proxy) => proxy,
            _ => panic!("valid source ended before EOF"),
        }
    }

    fn load(source: &[u8], profile: CompatibilityProfile, size: usize) -> LoadResult {
        let mut proxy = begin(
            LuaString::from("pipeline-test"),
            CompilerConfig::new(profile),
        )
        .unwrap();
        for fragment in source.chunks(size) {
            match proxy.resume(LoadInput::Bytes(LuaString::from(fragment))) {
                LoadPoll::NeedInput(next) => proxy = next,
                LoadPoll::Failed(error) => return LoadResult::Failed(error),
                LoadPoll::Loaded(_) => panic!("loaded before EOF"),
            }
        }
        match proxy.resume(LoadInput::End) {
            LoadPoll::Loaded(chunk) => LoadResult::Loaded(chunk),
            LoadPoll::Failed(error) => LoadResult::Failed(error),
            LoadPoll::NeedInput(_) => panic!("EOF needs input"),
        }
    }

    #[test]
    fn parser_advances_while_compiler_is_blocked() {
        let (blocked_sender, blocked_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Mutex::new(release_receiver);
        let (parsed_sender, parsed_receiver) = mpsc::channel();
        let compile_count = AtomicUsize::new(0);
        let hooks = Hooks {
            callback: Some(Arc::new(move |stage| {
                if stage == Stage::Compiling && compile_count.fetch_add(1, Ordering::SeqCst) == 0 {
                    blocked_sender.send(()).unwrap();
                    release_receiver.lock().unwrap().recv_timeout(WAIT).unwrap();
                }
                if stage == Stage::Parsed {
                    parsed_sender.send(()).unwrap();
                }
            })),
        };
        let release = Release(Some(release_sender));
        let proxy = start(
            LuaString::from("pipeline-test"),
            CompilerConfig::default(),
            hooks,
        )
        .unwrap();
        let proxy = feed(Box::new(proxy), b"local x=1; ");
        blocked_receiver.recv_timeout(WAIT).unwrap();
        parsed_receiver.recv_timeout(WAIT).unwrap();
        let proxy = feed(proxy, b"x=x+1; ");
        // The compiler is still blocked on the first batch. The parser has
        // already finished the next batch on its separate worker.
        parsed_receiver.recv_timeout(WAIT).unwrap();
        drop(release);
        assert!(matches!(proxy.resume(LoadInput::End), LoadPoll::Loaded(_)));
    }

    #[test]
    fn raw_queue_applies_bounded_backpressure() {
        let (blocked_sender, blocked_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Mutex::new(release_receiver);
        let first = AtomicBool::new(true);
        let hooks = Hooks {
            callback: Some(Arc::new(move |stage| {
                if stage == Stage::Parsing && first.swap(false, Ordering::SeqCst) {
                    blocked_sender.send(()).unwrap();
                    release_receiver.lock().unwrap().recv_timeout(WAIT).unwrap();
                }
            })),
        };
        let release = Release(Some(release_sender));
        let proxy = start(
            LuaString::from("pipeline-test"),
            CompilerConfig::default(),
            hooks,
        )
        .unwrap();
        let input = proxy.input.as_ref().unwrap();
        input
            .send(LoadInput::Bytes(LuaString::from("local x=1; ")))
            .unwrap();
        blocked_receiver.recv_timeout(WAIT).unwrap();
        for _ in 0..QUEUE_CAPACITY {
            input
                .try_send(LoadInput::Bytes(LuaString::from("x=x+1; ")))
                .unwrap();
        }
        assert!(matches!(
            input.try_send(LoadInput::Bytes(LuaString::from("x=x+1; "))),
            Err(mpsc::TrySendError::Full(_))
        ));
        drop(release);
        drop(proxy);
    }

    #[test]
    fn full_statement_queue_stops_parser_and_then_reader() {
        let (blocked_sender, blocked_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Mutex::new(release_receiver);
        let (parsed_sender, parsed_receiver) = mpsc::channel();
        let first = AtomicBool::new(true);
        let hooks = Hooks {
            callback: Some(Arc::new(move |stage| {
                if stage == Stage::Compiling && first.swap(false, Ordering::SeqCst) {
                    blocked_sender.send(()).unwrap();
                    release_receiver.lock().unwrap().recv_timeout(WAIT).unwrap();
                }
                if stage == Stage::Parsed {
                    parsed_sender.send(()).unwrap();
                }
            })),
        };
        let proxy = start(
            LuaString::from("pipeline-test"),
            CompilerConfig::default(),
            hooks,
        )
        .unwrap();
        let release = Release(Some(release_sender));
        let input = proxy.input.as_ref().unwrap();
        input
            .send(LoadInput::Bytes(LuaString::from("local x=1; ")))
            .unwrap();
        blocked_receiver.recv_timeout(WAIT).unwrap();
        parsed_receiver.recv_timeout(WAIT).unwrap();
        // Two batches queue behind the blocked compiler. The next completed
        // batch blocks its parser send, so raw input then fills its own queue.
        for _ in 0..=QUEUE_CAPACITY {
            input
                .send(LoadInput::Bytes(LuaString::from("x=x+1; ")))
                .unwrap();
            parsed_receiver.recv_timeout(WAIT).unwrap();
        }
        for _ in 0..QUEUE_CAPACITY {
            input
                .try_send(LoadInput::Bytes(LuaString::from("x=x+1; ")))
                .unwrap();
        }
        assert!(matches!(
            input.try_send(LoadInput::Bytes(LuaString::from("x=x+1; "))),
            Err(mpsc::TrySendError::Full(_))
        ));
        drop(release);
        assert!(matches!(
            Box::new(proxy).resume(LoadInput::End),
            LoadPoll::Loaded(_)
        ));
    }

    #[test]
    fn drop_cancels_and_joins_both_workers() {
        let exits = Arc::new(AtomicUsize::new(0));
        let observed = exits.clone();
        let hooks = Hooks {
            callback: Some(Arc::new(move |stage| {
                if matches!(stage, Stage::ParserExit | Stage::CompilerExit) {
                    observed.fetch_add(1, Ordering::SeqCst);
                }
            })),
        };
        let proxy = start(
            LuaString::from("pipeline-test"),
            CompilerConfig::default(),
            hooks,
        )
        .unwrap();
        drop(proxy);
        assert_eq!(exits.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn worker_panics_close_queues_and_join_both_workers() {
        for panic_stage in [Stage::Parsing, Stage::Compiling] {
            let exits = Arc::new(AtomicUsize::new(0));
            let observed = exits.clone();
            let hooks = Hooks {
                callback: Some(Arc::new(move |stage| {
                    if matches!(stage, Stage::ParserExit | Stage::CompilerExit) {
                        observed.fetch_add(1, Ordering::SeqCst);
                    }
                    assert_ne!(stage, panic_stage, "injected worker panic");
                })),
            };
            let proxy = start(
                LuaString::from("pipeline-test"),
                CompilerConfig::default(),
                hooks,
            )
            .unwrap();
            proxy
                .input
                .as_ref()
                .unwrap()
                .send(LoadInput::Bytes(LuaString::from("local x=1; ")))
                .unwrap();
            let result = proxy.result.recv_timeout(WAIT).unwrap();
            assert!(matches!(result, LoadResult::Failed(_)));
            drop(proxy);
            assert_eq!(exits.load(Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn syntax_failure_stops_parser_before_eof() {
        let (exit_sender, exit_receiver) = mpsc::channel();
        let hooks = Hooks {
            callback: Some(Arc::new(move |stage| {
                if stage == Stage::ParserExit {
                    exit_sender.send(()).unwrap();
                }
            })),
        };
        let mut proxy = start(
            LuaString::from("pipeline-test"),
            CompilerConfig::default(),
            hooks,
        )
        .unwrap();
        proxy
            .input
            .as_ref()
            .unwrap()
            .send(LoadInput::Bytes(LuaString::from("local = ")))
            .unwrap();
        exit_receiver.recv_timeout(WAIT).unwrap();
        let result = proxy.result.recv_timeout(WAIT).unwrap();
        assert!(matches!(result, LoadResult::Failed(_)));
        assert!(
            proxy
                .input
                .as_ref()
                .unwrap()
                .send(LoadInput::Bytes(LuaString::from("never parsed")))
                .is_err()
        );
        // A disconnected terminal result channel remains a terminal poll.
        assert!(proxy.poll().is_some());
    }

    #[test]
    fn eof_matches_sequential_chunks_for_all_profiles_and_fragment_sizes() {
        let source =
            b"local x=1\r\nlocal function f(y) return x+y end\nfor i=1,3 do x=f(i) end\rreturn x";
        for profile in CompatibilityProfile::ALL {
            let expected =
                compile_named("pipeline-test", source, CompilerConfig::new(profile)).unwrap();
            for size in [1, 2, 7, 17, source.len()] {
                match load(source, profile, size) {
                    LoadResult::Loaded(chunk) => {
                        assert_eq!(chunk, expected, "{profile:?}, size={size}");
                    }
                    LoadResult::Failed(error) => {
                        panic!("{}", String::from_utf8_lossy(error.as_bytes()))
                    }
                }
            }
        }
    }

    #[test]
    fn eof_preserves_closure_replay_labels_globals_and_deep_trees() {
        let deep = format!(
            "local x=1; value={}; return x",
            (0..128).map(|_| "1").collect::<Vec<_>>().join("+")
        );
        let numbers = (0..110)
            .map(|number| number.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let compact =
            format!("local wide=select('#',{numbers},select('#',{numbers})); return wide");
        for source in [
            b"local x=1; sink(x, mutate()); local function f() x=2 end; return f".as_slice(),
            b"goto done; local x=1; ::done:: ::last::",
            b"global<const> x=1; local function f() return x end; return f()",
            deep.as_bytes(),
            compact.as_bytes(),
        ] {
            for profile in CompatibilityProfile::ALL {
                let config = CompilerConfig::new(profile);
                let Ok(expected) = compile_named("pipeline-test", source, config) else {
                    continue;
                };
                for size in [1, 7, 127, source.len()] {
                    match load(source, profile, size) {
                        LoadResult::Loaded(chunk) => {
                            assert_eq!(chunk, expected, "{profile:?}, size={size}");
                        }
                        LoadResult::Failed(error) => {
                            panic!("{}", String::from_utf8_lossy(error.as_bytes()))
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn syntax_priority_and_semantic_diagnostics_match_sequential() {
        for source in [
            b"local x<const> =1; x=2; local =".as_slice(),
            b"local x<const> =1; x=2",
            b"local function f(...args) return args end; value=1",
            b"goto missing",
        ] {
            for profile in CompatibilityProfile::ALL {
                let config = CompilerConfig::new(profile);
                let name = LuaString::from("pipeline-test");
                let expected = match CompilerLoader::new(config)
                    .load_complete(name, LuaString::from(source))
                {
                    LoadPoll::Failed(error) => error,
                    LoadPoll::Loaded(_) => continue,
                    LoadPoll::NeedInput(_) => panic!("complete loader needs input"),
                };
                for size in [1, 7, source.len()] {
                    match load(source, profile, size) {
                        LoadResult::Failed(error) => {
                            assert_eq!(error, expected, "{profile:?}, size={size}");
                        }
                        LoadResult::Loaded(_) => panic!("invalid source loaded"),
                    }
                }
            }
        }
    }
}
