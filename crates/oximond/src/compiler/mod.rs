#![forbid(unsafe_code)]

//! Direct compiler for the first executable Lua subset.
//!
//! Locals have registers. Names not found in a lexical scope are globals. The
//! emitter captures every indexed target's table and key, then materializes
//! every assignment RHS before writing a target. That order is the part of
//! Lua assignment that is easy to get subtly wrong.

mod cache;
mod data;
mod locals;
#[cfg(not(target_family = "wasm"))]
mod pipeline;

use core::fmt;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::bytecode::load::{ChunkLoad, ChunkLoader, LoadInput, LoadPoll};
use crate::bytecode::{
    CallSite, Chunk, Constant, DebugMetadata, DiagnosticSite, Instruction, LocalVariable,
    UpvalueDescriptor, UpvalueSource, ValidationError,
};
use crate::compat::CompatibilityProfile;
use crate::string::LuaString;
use crate::syntax::{self as syntax, BinaryOp, Expr, Name, Span, Statement, TableField};

#[derive(Debug)]
pub enum CompileError {
    Syntax(syntax::SyntaxError),
    RegisterExhausted {
        span: Span,
    },
    ConstantPoolExhausted {
        span: Span,
    },
    UpvalueExhausted {
        span: Span,
    },
    InvalidBytecode(ValidationError),
    JumpOutOfRange {
        span: Span,
    },
    BreakOutsideLoop {
        span: Span,
    },
    AssignToReadOnly {
        span: Span,
        name: String,
    },
    UndeclaredGlobal {
        span: Span,
        name: String,
    },
    UndefinedLabel {
        span: Span,
        name: String,
    },
    DuplicateLabel {
        span: Span,
        name: String,
    },
    GotoIntoLocalScope {
        span: Span,
        name: String,
        local: String,
    },
}

/// Source-compilation settings at the host boundary.
///
/// The profile gates source syntax and keeps direct and loader-backed
/// compilation on the same Lua generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CompilerConfig {
    compatibility: CompatibilityProfile,
}

impl CompilerConfig {
    #[must_use]
    pub const fn new(compatibility: CompatibilityProfile) -> Self {
        Self { compatibility }
    }

    #[must_use]
    pub const fn compatibility_profile(self) -> CompatibilityProfile {
        self.compatibility
    }
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self::new(CompatibilityProfile::default())
    }
}
impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(error) => error.fmt(f),
            Self::RegisterExhausted { span } => {
                write!(f, "too many live compiler registers at byte {}", span.start)
            }
            Self::ConstantPoolExhausted { span } => {
                write!(f, "too many constants at byte {}", span.start)
            }
            Self::UpvalueExhausted { span } => {
                write!(f, "too many captured locals at byte {}", span.start)
            }
            Self::InvalidBytecode(error) => write!(f, "compiler emitted invalid bytecode: {error}"),
            Self::JumpOutOfRange { span } => {
                write!(f, "jump is out of range at byte {}", span.start)
            }
            Self::BreakOutsideLoop { span } => {
                write!(f, "break outside a loop at byte {}", span.start)
            }
            Self::AssignToReadOnly { span, name } => {
                write!(
                    f,
                    "attempt to assign to const variable '{name}' at byte {}",
                    span.start
                )
            }
            Self::UndeclaredGlobal { span, name } => {
                write!(
                    f,
                    "undeclared global variable '{name}' at byte {}",
                    span.start
                )
            }
            Self::UndefinedLabel { span, name } => {
                write!(
                    f,
                    "no visible label '{name}' for goto at byte {}",
                    span.start
                )
            }
            Self::DuplicateLabel { span, name } => {
                write!(
                    f,
                    "label '{name}' already defined in this block at byte {}",
                    span.start
                )
            }
            Self::GotoIntoLocalScope { span, name, local } => {
                write!(
                    f,
                    "goto '{name}' jumps into scope of '{local}' (local '{local}') at byte {}",
                    span.start
                )
            }
        }
    }
}
impl std::error::Error for CompileError {}
impl CompileError {
    fn span(&self) -> Option<Span> {
        match self {
            Self::Syntax(error) => Some(error.span),
            Self::RegisterExhausted { span }
            | Self::ConstantPoolExhausted { span }
            | Self::UpvalueExhausted { span }
            | Self::JumpOutOfRange { span }
            | Self::BreakOutsideLoop { span }
            | Self::AssignToReadOnly { span, .. }
            | Self::UndeclaredGlobal { span, .. }
            | Self::UndefinedLabel { span, .. }
            | Self::DuplicateLabel { span, .. }
            | Self::GotoIntoLocalScope { span, .. } => Some(*span),
            Self::InvalidBytecode(_) => None,
        }
    }

    fn diagnostic(&self, source_name: &LuaString, source: &[u8]) -> LuaString {
        let Some(span) = self.span() else {
            return LuaString::from(self.to_string());
        };
        let rendered = self.to_string();
        let suffix = format!(" at byte {}", span.start);
        let message = rendered.strip_suffix(&suffix).unwrap_or(&rendered);
        syntax::source_diagnostic(source_name, source, message, span)
    }
}
impl From<syntax::SyntaxError> for CompileError {
    fn from(error: syntax::SyntaxError) -> Self {
        Self::Syntax(error)
    }
}

/// Compiles Lua source with the default configuration and source name `=?`.
///
/// # Errors
///
/// Returns a syntax, capacity, or bytecode-validation error.
pub fn compile(source: impl AsRef<[u8]>) -> Result<Chunk, CompileError> {
    compile_named("=?", source, CompilerConfig::default())
}

/// Compiles Lua source with an explicit configuration and source name `=?`.
///
/// # Errors
///
/// Returns a syntax, capacity, or bytecode-validation error.
pub fn compile_with_config(
    source: impl AsRef<[u8]>,
    config: CompilerConfig,
) -> Result<Chunk, CompileError> {
    compile_named("=?", source, config)
}

/// Compiles Lua source with a source name for diagnostics and debug info.
///
/// # Errors
///
/// Returns a syntax, capacity, or bytecode-validation error.
pub fn compile_named(
    source_name: impl Into<LuaString>,
    source: impl AsRef<[u8]>,
    config: CompilerConfig,
) -> Result<Chunk, CompileError> {
    let source_name = source_name.into();
    let source = source.as_ref();
    let parser_config = syntax::ParserConfig::new(config.compatibility_profile());
    let (parsed, depth) = syntax::parse_bytes_with_depth(source, parser_config)
        .map_err(|error| CompileError::Syntax(error.with_source(&source_name, source)))?;
    compile_ast(source_name, parsed, depth, config)
}

/// Syntax trees up to this depth compile on the caller's stack. The
/// compiler recurses once per level, and unoptimized builds spend tens of
/// KiB per level, so this stays within a 2 MiB thread stack.
const CALLER_STACK_DEPTH: usize = 32;
/// Stack reserved per tree level when a deeper tree moves to its own thread.
/// Only touched pages are committed, so the figures are generous.
const STACK_BYTES_PER_LEVEL: usize = if cfg!(debug_assertions) {
    64 * 1024
} else {
    8 * 1024
};
const COMPILE_THREAD_BASE_STACK_BYTES: usize = 1024 * 1024;

/// Lowers a parsed source chunk into verified bytecode, then drops the tree.
fn compile_ast(
    source_name: LuaString,
    source: syntax::Chunk,
    depth: Option<usize>,
    config: CompilerConfig,
) -> Result<Chunk, CompileError> {
    let chunk = lower_ast(source_name, source, depth, config)?;
    chunk.validate().map_err(CompileError::InvalidBytecode)?;
    Ok(chunk)
}

/// Lowers a parsed source chunk into bytecode that is not validated yet, then
/// drops the tree.
///
/// Both steps recurse once per tree level. Lua nests blocks and delimiters up
/// to 200 levels and chains such as `a.b.c...` or `1 + 1 + ...` without
/// limit, so a deep tree gets a thread with a stack sized for it.
///
/// `depth` is the tree's `nesting_depth` when the parser already counted it,
/// which saves a walk over the whole tree.
fn lower_ast(
    source_name: LuaString,
    mut source: syntax::Chunk,
    depth: Option<usize>,
    config: CompilerConfig,
) -> Result<Chunk, CompileError> {
    // The stack size depends on this figure, so a parser that counted too
    // low would overflow it.
    debug_assert!(depth.is_none_or(|depth| depth >= source.nesting_depth()));
    let depth = depth.unwrap_or_else(|| source.nesting_depth());
    if depth <= CALLER_STACK_DEPTH {
        return compile_tree(source_name, &source, config);
    }
    let stack_bytes = depth
        .saturating_mul(STACK_BYTES_PER_LEVEL)
        .saturating_add(COMPILE_THREAD_BASE_STACK_BYTES);
    let tree = &mut source;
    let name = &source_name;
    let compiled = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("oximond-compile".to_owned())
            .stack_size(stack_bytes)
            .spawn_scoped(scope, move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compile_tree(name.clone(), tree, config)
                }));
                // Keep deep-tree destruction on this stack even if lowering
                // panics. The caller receives the panic after cleanup finishes.
                drop(core::mem::take(&mut tree.statements));
                result.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .map(|worker| {
                worker
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
    });
    // Without threads, as on some WebAssembly targets, the caller's stack
    // is all there is.
    compiled.unwrap_or_else(|_| compile_tree(source_name, &source, config))
}

fn compile_tree(
    source_name: LuaString,
    source: &syntax::Chunk,
    config: CompilerConfig,
) -> Result<Chunk, CompileError> {
    // As for nested functions, compact allocation is the fallback only.
    match compile_main(source_name.clone(), source, config, false) {
        Err(CompileError::RegisterExhausted { .. }) => {
            compile_main(source_name, source, config, true)
        }
        result => result,
    }
}

fn compile_main(
    source_name: LuaString,
    source: &syntax::Chunk,
    config: CompilerConfig,
    compact_registers: bool,
) -> Result<Chunk, CompileError> {
    let analysis = AnalysisCache::default();
    let mut compiler = new_main_compiler(
        source_name,
        Rc::new(source.line_starts.clone()),
        config,
        compact_registers,
        FreeNameAnalyzer::closure_assignments(&source.statements, &analysis),
        analysis,
    );
    compiler.push_scope();
    compiler.bind_implicit_environment()?;
    compiler.statements(&source.statements)?;
    finish_main(compiler, source.statements.last().map(statement_span))
}

fn new_main_compiler(
    source_name: LuaString,
    line_starts: Rc<Vec<usize>>,
    config: CompilerConfig,
    compact_registers: bool,
    closure_assigned_names: BTreeSet<Ident>,
    analysis: AnalysisCache,
) -> Compiler {
    let compatibility = config.compatibility_profile();
    // Lua treats the main chunk as a vararg function. It has no compatibility
    // `arg` table, but a top-level `...` remains legal.
    Compiler {
        is_vararg: true,
        compact_registers,
        source_name: Some(source_name),
        line_starts,
        readonly_control_variables: matches!(compatibility, CompatibilityProfile::Lua55),
        strict_label_shadowing: matches!(
            compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ),
        // LuaJIT 2.1 has Lua 5.2's goto and labels.
        goto_labels: matches!(
            compatibility,
            CompatibilityProfile::Lua52
                | CompatibilityProfile::Lua53
                | CompatibilityProfile::Lua54
                | CompatibilityProfile::Lua55
                | CompatibilityProfile::LuaJit21
        ),
        upvalue_limit: if compatibility.is_lua51_family() {
            60
        } else {
            255
        },
        legacy_vararg_compat: compatibility.is_lua51_family(),
        lexical_environment: matches!(
            compatibility,
            CompatibilityProfile::Lua52
                | CompatibilityProfile::Lua53
                | CompatibilityProfile::Lua54
                | CompatibilityProfile::Lua55
        ),
        tagged_integers: compatibility.is_lua53_or_later(),
        repeat_header_line: matches!(compatibility, CompatibilityProfile::LuaJit21),
        generic_for_exit_line: matches!(
            compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ),
        field_key_names: match compatibility {
            CompatibilityProfile::LuaJit21 => FieldKeyNames::Unnamed,
            CompatibilityProfile::Lua51
            | CompatibilityProfile::Lua52
            | CompatibilityProfile::Lua53 => FieldKeyNames::Unknown,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => {
                FieldKeyNames::IntegerIndex
            }
        },
        constant_names: matches!(
            compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ),
        subtract_immediate: matches!(
            compatibility,
            CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
        ),
        closure_assigned_names: Rc::new(closure_assigned_names),
        analysis,
        ..Compiler::default()
    }
}

fn finish_main(
    mut compiler: Compiler,
    last_statement: Option<Span>,
) -> Result<Chunk, CompileError> {
    compiler.resolve_gotos()?;
    compiler.close_all_scopes();
    let terminal_pc = compiler.code.len();
    compiler.code.push(Instruction::halt());
    if let Some(span) = last_statement {
        compiler.instruction_lines.insert(
            terminal_pc,
            compiler.line_for(Span {
                start: span.end.saturating_sub(1),
                end: span.end,
            }),
        );
    }
    let line_info = compiler.line_info();
    let local_variables = compiler.take_debug_locals();
    let chunk = Chunk::with_varargs(
        compiler.register_count.max(1),
        0,
        compiler.constants,
        compiler.code,
        compiler.prototypes,
        compiler.upvalues,
        None,
    )
    .with_debug_metadata(
        DebugMetadata::new(
            compiler
                .source_name
                .unwrap_or_else(|| LuaString::from("=?")),
            compiler.call_sites,
        )
        .with_diagnostic_sites(compiler.diagnostic_sites)
        .with_local_variables(local_variables)
        .with_line_info(line_info),
    );
    // The caller validates the chunk. The live register analysis runs on the
    // first call (see `Chunk::entry_register_count`): most loaded functions
    // never run.
    Ok(chunk)
}

/// A neutral loader adapter that parses byte fragments as they arrive.
///
/// Complete top-level statements compile as fragments arrive. The chunk is
/// validated and returned only after the parser accepts the end marker.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct CompilerLoader {
    config: CompilerConfig,
}

impl CompilerLoader {
    /// Creates a loader carrying an explicit compatibility configuration.
    #[must_use]
    pub const fn new(config: CompilerConfig) -> Self {
        Self { config }
    }

    /// Creates a loader for one selected compatibility profile.
    #[must_use]
    pub const fn with_profile(compatibility: CompatibilityProfile) -> Self {
        Self::new(CompilerConfig::new(compatibility))
    }

    #[must_use]
    pub const fn config(self) -> CompilerConfig {
        self.config
    }

    /// Compiles `source` when the whole-source parser accepts it and lowering succeeds.
    ///
    /// `None` leaves every other case, including syntax errors and sources that
    /// continue past `source`, to [`ChunkLoader::begin`], whose diagnostics and
    /// incremental behaviour are unchanged. A caller holding the first piece of a
    /// reader's input uses this to skip the streaming parser when that piece turns out
    /// to be the whole chunk.
    ///
    /// The chunk is not validated here: the caller installs it with
    /// [`crate::vm::Vm::load_chunk`], which validates every chunk it is given,
    /// so validating here as well would check each source twice.
    ///
    /// A chunk that a Lua state of the process compiled before is usually not
    /// compiled again: see [`cache`] for which chunks it keeps.
    #[must_use]
    pub fn try_load_complete(self, source_name: &LuaString, source: &[u8]) -> Option<Chunk> {
        self.try_load_piece(source_name, source, || true)
    }

    /// [`Self::try_load_complete`] for the first piece of a reader's input,
    /// which may turn out to be only the start of the chunk. `ended` runs once
    /// the chunk is in hand and says whether the input ended after `source`;
    /// the cache keeps the chunk only then.
    #[must_use]
    pub fn try_load_piece(
        self,
        source_name: &LuaString,
        source: &[u8],
        ended: impl FnOnce() -> bool,
    ) -> Option<Chunk> {
        cache::load(
            self.config,
            source_name,
            source,
            || self.try_load_complete_uncached(source_name, source),
            ended,
        )
    }

    fn try_load_complete_uncached(self, source_name: &LuaString, source: &[u8]) -> Option<Chunk> {
        data::compile(source_name, source, self.config)
            .or_else(|| self.compile_tree(source_name, source))
    }

    /// [`Self::try_load_complete_uncached`] through the syntax tree alone.
    fn compile_tree(self, source_name: &LuaString, source: &[u8]) -> Option<Chunk> {
        let parser_config = syntax::ParserConfig::new(self.config.compatibility_profile());
        let (parsed, depth) = syntax::parse_complete_only(source, parser_config)?;
        lower_ast(source_name.clone(), parsed, Some(depth), self.config).ok()
    }
}

impl ChunkLoader for CompilerLoader {
    fn begin(&self, source_name: LuaString) -> Box<dyn ChunkLoad> {
        Box::new(CompilerLoad {
            source_name,
            config: self.config,
            parser: syntax::StreamingParser::with_profile(self.config.compatibility_profile()),
            source: Vec::new(),
            compilation: IncrementalCompilation::default(),
        })
    }

    #[cfg(not(target_family = "wasm"))]
    fn begin_pipelined(&self, source_name: LuaString) -> Option<Box<dyn ChunkLoad>> {
        pipeline::begin(source_name, self.config)
    }

    fn load_complete(&self, source_name: LuaString, source: LuaString) -> LoadPoll {
        let source = source.as_bytes();
        let parser_config = syntax::ParserConfig::new(self.config.compatibility_profile());
        match syntax::parse_bytes_with_depth(source, parser_config) {
            Ok((parsed, depth)) => {
                match compile_ast(source_name.clone(), parsed, depth, self.config) {
                    Ok(chunk) => LoadPoll::Loaded(chunk),
                    Err(error) => LoadPoll::Failed(error.diagnostic(&source_name, source)),
                }
            }
            Err(error) => LoadPoll::Failed(error.with_source(&source_name, source).diagnostic()),
        }
    }
}

struct CompilerLoad {
    source_name: LuaString,
    config: CompilerConfig,
    parser: syntax::StreamingParser,
    source: Vec<u8>,
    compilation: IncrementalCompilation,
}

impl ChunkLoad for CompilerLoad {
    fn resume(mut self: Box<Self>, input: LoadInput) -> LoadPoll {
        match input {
            LoadInput::Bytes(bytes) => {
                self.source.extend_from_slice(bytes.as_bytes());
                let poll = self.parser.push(bytes.as_bytes());
                if matches!(poll, syntax::SourceParsePoll::NeedInput) {
                    self.compilation.advance(
                        &self.source_name,
                        self.config,
                        self.parser.completed_statements(),
                        self.parser.line_starts(),
                        false,
                    );
                }
                self.poll_parser(poll)
            }
            LoadInput::End => match self.parser.end_owned() {
                Ok(source) => {
                    match self
                        .compilation
                        .finish(self.source_name.clone(), source, self.config)
                    {
                        Ok(chunk) => LoadPoll::Loaded(chunk),
                        Err(error) => {
                            LoadPoll::Failed(error.diagnostic(&self.source_name, &self.source))
                        }
                    }
                }
                Err(error) => LoadPoll::Failed(
                    error
                        .with_source(&self.source_name, &self.source)
                        .diagnostic(),
                ),
            },
        }
    }
}

impl CompilerLoad {
    fn poll_parser(self: Box<Self>, poll: syntax::SourceParsePoll) -> LoadPoll {
        match poll {
            syntax::SourceParsePoll::NeedInput => LoadPoll::NeedInput(self),
            syntax::SourceParsePoll::Parsed(source) => {
                match self
                    .compilation
                    .finish(self.source_name.clone(), source, self.config)
                {
                    Ok(chunk) => LoadPoll::Loaded(chunk),
                    Err(error) => {
                        LoadPoll::Failed(error.diagnostic(&self.source_name, &self.source))
                    }
                }
            }
            syntax::SourceParsePoll::Failed(error) => LoadPoll::Failed(
                error
                    .with_source(&self.source_name, &self.source)
                    .diagnostic(),
            ),
        }
    }
}

/// Keeps lowering state between parser pulls. The parser retains the tree for
/// the uncommon cases that require the existing whole-function retry.
#[derive(Default)]
struct IncrementalCompilation {
    compiler: Option<Compiler>,
    emitted: usize,
    analyzed: usize,
    replay: bool,
    deferred_error: Option<CompileError>,
    analysis: AnalysisCache,
}

impl IncrementalCompilation {
    fn advance(
        &mut self,
        source_name: &LuaString,
        config: CompilerConfig,
        statements: &[Statement],
        line_starts: &[usize],
        end: bool,
    ) {
        if self.deferred_error.is_some() || self.replay {
            return;
        }
        // Recursive lowering and tree analysis need the sized-stack path for
        // deep statements. Check depth before either visitor sees the tree.
        if statements[self.analyzed..]
            .iter()
            .any(|statement| statement.nesting_depth() > CALLER_STACK_DEPTH)
        {
            self.replay = true;
            return;
        }
        // The parser may have moved the statements since the last call, and
        // the cache is keyed by the address of a function body.
        self.analysis.clear();
        let newly_assigned =
            FreeNameAnalyzer::closure_assignments(&statements[self.analyzed..], &self.analysis);
        let analysis = &self.analysis;
        let compiler = self.compiler.get_or_insert_with(|| {
            new_main_compiler(
                source_name.clone(),
                Rc::default(),
                config,
                false,
                BTreeSet::new(),
                analysis.clone(),
            )
        });
        if self.emitted > 0 && !newly_assigned.is_subset(&compiler.closure_assigned_names) {
            // Future closure writes can change copies emitted for earlier calls
            // and assignments. Replay once with the final whole-unit name set.
            self.replay = true;
            return;
        }
        Rc::make_mut(&mut compiler.closure_assigned_names).extend(newly_assigned);
        self.analyzed = statements.len();
        // Only the last known physical line start can change when a CRLF or
        // LFCR pair is split across pulls. Keep the stable prefix in place.
        let stable = compiler
            .line_starts
            .len()
            .saturating_sub(1)
            .min(line_starts.len());
        let known = Rc::make_mut(&mut compiler.line_starts);
        known.truncate(stable);
        known.extend_from_slice(&line_starts[stable..]);
        if compiler.scopes.is_empty() {
            compiler.push_scope();
            if let Err(error) = compiler.bind_implicit_environment() {
                self.deferred_error = Some(error);
                return;
            }
        }
        let ready = if end {
            statements.len()
        } else {
            statements
                .iter()
                .rposition(|statement| !matches!(statement, Statement::Label { .. }))
                .map_or(0, |index| index + 1)
        };
        for index in self.emitted..ready {
            let trailing_label = end
                && matches!(statements[index], Statement::Label { .. })
                && statements[index + 1..]
                    .iter()
                    .all(|statement| matches!(statement, Statement::Label { .. }));
            if let Err(error) = compiler.statement(&statements[index], trailing_label) {
                if matches!(error, CompileError::RegisterExhausted { .. }) {
                    self.replay = true;
                } else {
                    // A later syntax error takes precedence over lowering errors.
                    self.deferred_error = Some(error);
                }
                return;
            }
            self.emitted = index + 1;
        }
    }

    fn finish(
        mut self,
        source_name: LuaString,
        source: syntax::Chunk,
        config: CompilerConfig,
    ) -> Result<Chunk, CompileError> {
        // Even an earlier semantic error leaves the parser owning later
        // statements. Drop a deep final tree on the existing sized stack.
        if self.deferred_error.is_some() && source.nesting_depth() > CALLER_STACK_DEPTH {
            drop(self.compiler.take());
            return compile_ast(source_name, source, None, config);
        }
        self.advance(
            &source_name,
            config,
            &source.statements,
            &source.line_starts,
            true,
        );
        if self.replay {
            drop(self.compiler.take());
            return compile_ast(source_name, source, None, config);
        }
        if let Some(error) = self.deferred_error {
            return Err(error);
        }
        let chunk = finish_main(
            self.compiler.expect("end starts even an empty compiler"),
            source.statements.last().map(statement_span),
        )?;
        chunk.validate().map_err(CompileError::InvalidBytecode)?;
        Ok(chunk)
    }
}

type Register = u8;
/// A source name, shared with the syntax tree.
type Ident = std::sync::Arc<str>;
#[derive(Default)]
#[allow(clippy::struct_excessive_bools)] // Compatibility features are independent compiler switches.
struct Compiler {
    constants: Vec<Constant>,
    /// Indices into `constants` of the constants that are not strings.
    constant_indices: rustc_hash::FxHashMap<ConstantKey, u32>,
    /// Indices into `constants` of the string constants, hashed by their
    /// bytes, so a lookup needs no `LuaString`.
    string_indices: hashbrown::HashTable<u32>,
    code: Vec<Instruction>,
    next_register: u8,
    register_count: u8,
    scopes: locals::LocalScopes,
    readonly_scopes: Vec<BTreeSet<Ident>>,
    /// Per scope, the `<const>` locals bound to an integer constant, which
    /// Lua folds. Declaring a name again in the scope drops its entry.
    constant_scopes: Vec<BTreeMap<Ident, i64>>,
    /// The integer `<const>` locals of enclosing functions that are visible
    /// here.
    inherited_constants: BTreeMap<Ident, i64>,
    close_scopes: Vec<Vec<Register>>,
    global_scopes: Vec<GlobalScope>,
    global_barrier_scopes: Vec<Vec<(usize, String)>>,
    next_global_barrier: usize,
    debug_local_scopes: Vec<Vec<OpenLocal>>,
    debug_locals: Vec<(usize, LocalVariable)>,
    next_debug_local: usize,
    prototypes: Vec<Chunk>,
    parameter_count: u8,
    upvalues: Vec<UpvalueDescriptor>,
    upvalue_limit: usize,
    upvalue_names: BTreeMap<Ident, u16>,
    /// Where the names this function may take from its parent live there:
    /// the names free in the function, and `_ENV`.
    parent_captures: rustc_hash::FxHashMap<Ident, UpvalueSource>,
    captured_registers: locals::RegisterBits,
    reserved_register_floors: Vec<Register>,
    is_vararg: bool,
    arg_register: Option<Register>,
    legacy_vararg_compat: bool,
    loop_breaks: Vec<Vec<usize>>,
    loop_close_depths: Vec<usize>,
    labels: Vec<LabelInfo>,
    gotos: Vec<GotoInfo>,
    scope_path: Vec<u32>,
    next_scope_id: u32,
    source_name: Option<LuaString>,
    call_sites: Vec<CallSite>,
    diagnostic_sites: Vec<DiagnosticSite>,
    // Sites arrive in bytecode order. A duplicate can only be another operand
    // of the instruction currently being emitted, so a short ordered suffix
    // replaces a tree lookup for every arithmetic operation.
    diagnostic_site_keys: Vec<(usize, Register)>,
    /// Shared by every function of the unit.
    line_starts: Rc<Vec<usize>>,
    instruction_lines: InstructionLines,
    control_flow_depth: usize,
    readonly_control_variables: bool,
    goto_labels: bool,
    strict_label_shadowing: bool,
    lexical_environment: bool,
    tagged_integers: bool,
    /// `LuaJIT` emits a loop marker at the `repeat` keyword on each iteration.
    repeat_header_line: bool,
    /// Lua 5.4 and later close the generic-for scope on the `end` line.
    generic_for_exit_line: bool,
    /// How errors name a field read through a key that is not a constant
    /// string.
    field_key_names: FieldKeyNames,
    /// Whether errors name a string constant operand, as Lua 5.4 does.
    constant_names: bool,
    /// Whether `x - k` for a small integer constant `k` is an addition of
    /// `-k`, as in Lua 5.4's `OP_ADDI`, which makes `-0.0 - 0` positive.
    subtract_immediate: bool,
    has_global_declarations: bool,
    /// The highest `next_register` before scratch registers were released
    /// inside the current statement. Released registers may still hold dead
    /// values, so the statement boundary clears up to this mark.
    released_high_water: Register,
    /// Allocates registers like Lua's own compiler, releasing scratch
    /// registers inside expressions. Used only for a function that exhausts
    /// its registers otherwise; see `Compiler::function_body`.
    compact_registers: bool,
    /// Names that this function's nested closures assign; see
    /// `Compiler::call_operand`.
    closure_assigned_names: Rc<BTreeSet<Ident>>,
    /// Per-function analyses of the tree being compiled.
    analysis: AnalysisCache,
    /// String constants of the whole compilation unit, shared by every
    /// function compiled from it. Equal constants then share one string,
    /// so a field written by one function and read by another matches on
    /// identity instead of comparing bytes.
    string_constants: StringConstantPool,
}

/// One shared string per distinct string constant of a compilation unit.
#[derive(Clone, Default)]
struct StringConstantPool(Rc<RefCell<PoolEntries>>);

#[derive(Default)]
struct PoolEntries {
    strings: hashbrown::HashTable<LuaString>,
    /// The string of each source name seen so far, found by the address of
    /// its shared text, which the entry keeps alive.
    names: hashbrown::HashTable<(Ident, LuaString)>,
}

impl StringConstantPool {
    /// The unit's string with `bytes`, whose `cached_hash_bytes` is `hash`,
    /// built and recorded only when `bytes` is new to the unit.
    fn intern_bytes(&self, bytes: &[u8], hash: u64) -> LuaString {
        let pool = &mut self.0.borrow_mut().strings;
        if let Some(shared) = pool.find(hash, |shared| shared.as_bytes() == bytes) {
            return shared.clone();
        }
        let string = crate::string::constants::shared_bytes(bytes, hash);
        pool.insert_unique(hash, string.clone(), LuaString::cached_hash);
        string
    }

    /// The unit's string for the source name `name`. The whole-source
    /// parser shares one text per name, so most calls find it by address
    /// without hashing its bytes.
    fn name(&self, name: &Ident) -> LuaString {
        let address = |name: &Ident| {
            std::hash::BuildHasher::hash_one(
                &rustc_hash::FxBuildHasher,
                std::sync::Arc::as_ptr(name).cast::<u8>(),
            )
        };
        let hash = address(name);
        if let Some((_, string)) = self
            .0
            .borrow()
            .names
            .find(hash, |(shared, _)| std::sync::Arc::ptr_eq(shared, name))
        {
            return string.clone();
        }
        let bytes = name.as_bytes();
        let string = self.intern_bytes(bytes, crate::string::cached_hash_bytes(bytes));
        self.0.borrow_mut().names.insert_unique(
            hash,
            (name.clone(), string.clone()),
            |(shared, _)| address(shared),
        );
        string
    }

    /// The unit's string equal to `string`, recording `string` itself when
    /// it is new to the unit, so nothing is built.
    fn intern(&self, string: &LuaString) -> LuaString {
        let pool = &mut self.0.borrow_mut().strings;
        let hash = string.cached_hash();
        if let Some(shared) = pool.find(hash, |shared| shared == string) {
            return shared.clone();
        }
        let string = crate::string::constants::shared(string);
        pool.insert_unique(hash, string.clone(), LuaString::cached_hash);
        string
    }
}

#[derive(Clone)]
struct OpenLocal {
    order: usize,
    name: LuaString,
    register: Register,
    start_pc: usize,
}
#[derive(Clone, Default)]
#[allow(clippy::struct_excessive_bools)] // Lua 5.5 global declarations combine independent restrictions.
struct GlobalScope {
    names: BTreeMap<String, bool>,
    restricted: bool,
    allow_all: bool,
    all_readonly: bool,
    deny_all: bool,
}
/// How a call's argument list ends after its fixed arguments.
#[derive(Clone, Copy)]
enum CallTail {
    Fixed,
    /// The caller's varargs follow the fixed arguments.
    Vararg,
    /// An inner call's open results follow the fixed arguments.
    Open,
}

/// How errors name a field read through a key that is not a constant
/// string, following each reference's debug information.
#[derive(Clone, Copy, Default)]
enum FieldKeyNames {
    /// `LuaJIT` names no such field.
    #[default]
    Unnamed,
    /// Lua 5.1 through 5.3 call it field `'?'`.
    Unknown,
    /// Lua 5.4 and later also call it field `'?'`, except that a key from
    /// 0 to 255, read with `GETI`, is field `'integer index'`.
    IntegerIndex,
}

enum CapturedTarget {
    Local(Register),
    Upvalue(u16),
    Global(u32),
    /// A table field; `name` names the table in an error for indexing a
    /// value that is not one.
    Index {
        table: Register,
        key: Register,
        name: Option<(LuaString, crate::bytecode::CallKind)>,
    },
    /// A table field named by string constant `field`.
    Field {
        table: Register,
        field: u8,
        name: Option<(LuaString, crate::bytecode::CallKind)>,
    },
    /// A field named by string constant `field` of the table in `upvalue`.
    TabUp {
        upvalue: u8,
        field: u8,
    },
}

struct LabelInfo {
    name: String,
    pc: usize,
    scope_path: Vec<u32>,
    locals: BTreeMap<usize, (Register, String)>,
    close_locals: BTreeSet<Register>,
    global_barriers: BTreeMap<usize, String>,
}

struct GotoInfo {
    name: String,
    span: Span,
    instruction: usize,
    scope_path: Vec<u32>,
    locals: BTreeMap<usize, Register>,
    captured_locals: locals::RegisterBits,
    close_locals: Vec<Register>,
    global_barriers: BTreeSet<usize>,
}

/// The source line of each instruction, by pc. Zero means no line yet:
/// [`Compiler::line_for`] numbers lines from one. A dense vector, since
/// nearly every instruction gets a line.
#[derive(Clone, Debug, Default)]
struct InstructionLines(Vec<u32>);

impl InstructionLines {
    /// Sets the line of `pc`, replacing an earlier one.
    fn insert(&mut self, pc: usize, line: u32) {
        if pc >= self.0.len() {
            self.0.resize(pc + 1, 0);
        }
        self.0[pc] = line;
    }

    /// Sets the line of `pc` unless it has one.
    fn insert_if_absent(&mut self, pc: usize, line: u32) {
        if pc >= self.0.len() {
            self.0.resize(pc + 1, 0);
        }
        if self.0[pc] == 0 {
            self.0[pc] = line;
        }
    }

    /// The recorded lines, by pc, with zero for none.
    fn as_slice(&self) -> &[u32] {
        &self.0
    }
}

/// An ordered, bit-exact key for compiler constant interning. `f64` itself
/// lacks `Ord`, but bytecode constants need deterministic interning even in a
/// source file large enough to require wide references.
#[derive(Clone, Debug, Eq, PartialEq)]
enum ConstantKey {
    Nil,
    Boolean(bool),
    Integer(i64),
    Number(u64),
    Int64(i64),
    UInt64(u64),
}

/// One well-mixed word per key. `FxHasher` multiplies once, which leaves
/// the bits the map picks its control byte from nearly the same for keys
/// that differ only in their high or low bits (floats with short
/// mantissas, integers far apart), and the probes of a big data chunk's
/// constants then compare one key after another.
impl std::hash::Hash for ConstantKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let (tag, bits): (u64, u64) = match self {
            Self::Nil => (0, 0),
            Self::Boolean(value) => (1, u64::from(*value)),
            Self::Integer(value) => (2, value.cast_unsigned()),
            Self::Number(bits) => (3, *bits),
            Self::Int64(value) => (4, value.cast_unsigned()),
            Self::UInt64(value) => (5, *value),
        };
        // The finalizer of splitmix64.
        let mut word = bits ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        word = (word ^ (word >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        state.write_u64(word ^ (word >> 31));
    }
}

impl ConstantKey {
    /// The key of a constant that is not a string. Strings are indexed by
    /// their bytes in `Compiler::string_indices`.
    fn of(constant: &Constant) -> Option<Self> {
        Some(match constant {
            Constant::Nil => Self::Nil,
            Constant::Boolean(value) => Self::Boolean(*value),
            Constant::Integer(value) => Self::Integer(*value),
            Constant::Number(value) => Self::Number(value.to_bits()),
            Constant::String(_) => return None,
            Constant::Int64(value) => Self::Int64(*value),
            Constant::UInt64(value) => Self::UInt64(*value),
        })
    }
}

fn expands_open_call_results(expression: &Expr) -> bool {
    matches!(expression, Expr::Call { .. } | Expr::MethodCall { .. })
}

fn collect_concat_operands<'a>(expression: &'a Expr, operands: &mut Vec<&'a Expr>) {
    match expression {
        Expr::Binary {
            op: BinaryOp::Concat,
            left,
            right,
            ..
        } => {
            collect_concat_operands(left, operands);
            collect_concat_operands(right, operands);
        }
        // Parentheses retain their own expression node. They deliberately
        // prevent flattening, because they define an earlier concatenation
        // boundary for metamethods.
        expression => operands.push(expression),
    }
}

/// Results of the per-function tree analyses, shared by the compilers of
/// one unit. A nested function is analyzed for its parent's analysis and
/// again for its own lowering, and each level of nesting would repeat that
/// for every function inside it. Entries are keyed by the address of a
/// function body, so the cache is valid only while one tree is neither moved
/// nor dropped: `compile_main` makes a fresh one, and an incremental
/// compilation clears it before each batch of statements.
#[derive(Clone, Default)]
struct AnalysisCache(Rc<RefCell<AnalysisEntries>>);

#[derive(Default)]
struct AnalysisEntries {
    free_names: rustc_hash::FxHashMap<(usize, bool), Rc<Vec<Ident>>>,
    closure_assignments: rustc_hash::FxHashMap<usize, BodyAssignments>,
}

impl AnalysisCache {
    fn clear(&self) {
        let mut entries = self.0.borrow_mut();
        entries.free_names.clear();
        entries.closure_assignments.clear();
    }
}

/// Collects the names a function must obtain from an enclosing lexical scope.
///
/// A nested function contributes only names that its immediate parent does not
/// bind. That lets an intermediate closure forward an ancestor capture without
/// reserving a slot for every visible local.
///
/// The same walk collects the names that nested closures assign. Only such a
/// closure can change one of the function's locals while the function is
/// still evaluating an expression. That part ignores shadowing, so the set
/// may name more locals than a closure actually reaches.
#[derive(Default)]
struct FreeNameAnalyzer {
    cache: AnalysisCache,
    /// Whether to collect free names; a main chunk wants only assignments.
    collect_names: bool,
    names: Vec<Ident>,
    seen: rustc_hash::FxHashSet<Ident>,
    /// The names bound so far, innermost last; `scope_starts` holds where
    /// each open nested scope begins.
    bound: Vec<Ident>,
    scope_starts: Vec<usize>,
    /// Names assigned by the walked statements themselves.
    direct: BTreeSet<Ident>,
    /// Names assigned inside functions nested in the walked statements.
    nested: BTreeSet<Ident>,
}

impl FreeNameAnalyzer {
    /// The names assigned inside functions nested in `statements`.
    fn closure_assignments(statements: &[Statement], cache: &AnalysisCache) -> BTreeSet<Ident> {
        let mut analyzer = Self {
            cache: cache.clone(),
            ..Self::default()
        };
        analyzer.statements(statements);
        analyzer.nested
    }

    /// A function's free names and its body's assignments, analyzed once
    /// per body.
    fn function(
        method: bool,
        parameters: &[Name],
        vararg: bool,
        vararg_name: Option<&Name>,
        body: &syntax::Block,
        cache: &AnalysisCache,
    ) -> (Rc<Vec<Ident>>, BodyAssignments) {
        let body_key = std::ptr::from_ref(body) as usize;
        let key = (body_key, method);
        {
            let entries = cache.0.borrow();
            if let (Some(names), Some(assignments)) = (
                entries.free_names.get(&key),
                entries.closure_assignments.get(&body_key),
            ) {
                return (names.clone(), assignments.clone());
            }
        }
        let mut analyzer = Self {
            cache: cache.clone(),
            collect_names: true,
            ..Self::default()
        };
        if method {
            analyzer.bind(&Ident::from("self"));
        }
        for parameter in parameters {
            analyzer.bind(&parameter.text);
        }
        // Lua 5.1 gives every vararg function an implicit `arg` local. Lua
        // 5.5 instead optionally names the vararg table directly.
        if let Some(vararg_name) = vararg_name {
            analyzer.bind(&vararg_name.text);
        } else if vararg {
            analyzer.bind(&Ident::from("arg"));
        }
        analyzer.statements(&body.statements);
        let names = Rc::new(analyzer.names);
        let assignments = Rc::new((analyzer.direct, Rc::new(analyzer.nested)));
        let mut entries = cache.0.borrow_mut();
        entries.free_names.insert(key, names.clone());
        entries
            .closure_assignments
            .insert(body_key, assignments.clone());
        (names, assignments)
    }

    fn statements(&mut self, statements: &[Statement]) {
        for statement in statements {
            self.statement(statement);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Do(block) => self.scoped_block(block),
            Statement::Local { names, values, .. } => {
                for value in values {
                    self.expression(value);
                }
                for name in names {
                    self.bind(&name.text);
                }
            }
            Statement::Global { values, .. } | Statement::Return { values, .. } => {
                for value in values {
                    self.expression(value);
                }
            }
            Statement::Assign {
                targets, values, ..
            } => {
                for target in targets {
                    if let Expr::Name(name) = target {
                        self.assign(&name.text);
                    }
                    self.expression(target);
                }
                for value in values {
                    self.expression(value);
                }
            }
            Statement::Function {
                name,
                parameters,
                vararg,
                vararg_name,
                body,
                ..
            } => {
                if let Some(first) = name.segments.first() {
                    self.use_name(&first.text);
                }
                if let [only] = name.segments.as_slice()
                    && !name.method
                {
                    self.assign(&only.text);
                }
                self.forward_nested(*vararg, vararg_name.as_ref(), name.method, parameters, body);
            }
            Statement::LocalFunction {
                name,
                parameters,
                vararg,
                vararg_name,
                body,
                ..
            } => {
                // This declaration binds before its body, so recursion is
                // local and does not become an outer free name.
                self.bind(&name.text);
                self.forward_nested(*vararg, vararg_name.as_ref(), false, parameters, body);
            }
            Statement::If {
                condition,
                body,
                else_ifs,
                else_body,
                ..
            } => {
                self.expression(condition);
                self.scoped_block(body);
                for arm in else_ifs {
                    self.expression(&arm.condition);
                    self.scoped_block(&arm.body);
                }
                if let Some(body) = else_body {
                    self.scoped_block(body);
                }
            }
            Statement::While {
                condition, body, ..
            } => {
                self.expression(condition);
                self.scoped_block(body);
            }
            Statement::Repeat {
                body, condition, ..
            } => {
                self.push_scope();
                self.statements(&body.statements);
                self.expression(condition);
                self.pop_scope();
            }
            Statement::Break { .. } | Statement::Goto { .. } | Statement::Label { .. } => {}
            Statement::NumericFor {
                name,
                start,
                limit,
                step,
                body,
                ..
            } => {
                self.expression(start);
                self.expression(limit);
                if let Some(step) = step {
                    self.expression(step);
                }
                self.push_scope();
                self.bind(&name.text);
                self.statements(&body.statements);
                self.pop_scope();
            }
            Statement::GenericFor {
                names,
                expressions,
                body,
                ..
            } => {
                for expression in expressions {
                    self.expression(expression);
                }
                self.push_scope();
                for name in names {
                    self.bind(&name.text);
                }
                self.statements(&body.statements);
                self.pop_scope();
            }
            Statement::Call(expression) => self.expression(expression),
        }
    }

    fn scoped_block(&mut self, block: &syntax::Block) {
        self.push_scope();
        self.statements(&block.statements);
        self.pop_scope();
    }

    fn push_scope(&mut self) {
        self.scope_starts.push(self.bound.len());
    }

    fn pop_scope(&mut self) {
        let start = self.scope_starts.pop().expect("free-name scope is open");
        self.bound.truncate(start);
    }

    fn expression(&mut self, expression: &Expr) {
        match expression {
            Expr::Name(name) => self.use_name(&name.text),
            Expr::Table { fields, .. } => {
                for field in fields {
                    match field {
                        TableField::List(value) | TableField::Name { value, .. } => {
                            self.expression(value);
                        }
                        TableField::Key { key, value } => {
                            self.expression(key);
                            self.expression(value);
                        }
                    }
                }
            }
            Expr::Function {
                parameters,
                vararg,
                vararg_name,
                body,
                ..
            } => self.forward_nested(*vararg, vararg_name.as_deref(), false, parameters, body),
            Expr::Index { table, key, .. } => {
                self.expression(table);
                self.expression(key);
            }
            Expr::Binary { left, right, .. } => {
                self.expression(left);
                self.expression(right);
            }
            Expr::Unary { expression, .. } | Expr::Parenthesized { expression, .. } => {
                self.expression(expression);
            }
            Expr::Call {
                callee, arguments, ..
            } => {
                self.expression(callee);
                for argument in arguments {
                    self.expression(argument);
                }
            }
            Expr::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.expression(receiver);
                for argument in arguments {
                    self.expression(argument);
                }
            }
            Expr::Nil(_)
            | Expr::Boolean(_, _)
            | Expr::Integer(_, _)
            | Expr::IntegerBoundary(_)
            | Expr::Number(_, _)
            | Expr::Int64(_, _)
            | Expr::UInt64(_, _)
            | Expr::String(_, _)
            | Expr::Vararg(_) => {}
        }
    }

    fn forward_nested(
        &mut self,
        vararg: bool,
        vararg_name: Option<&Name>,
        method: bool,
        parameters: &[Name],
        body: &syntax::Block,
    ) {
        let cache = self.cache.clone();
        let (names, assignments) =
            Self::function(method, parameters, vararg, vararg_name, body, &cache);
        for name in names.iter() {
            self.use_name(name);
        }
        self.nested.extend(assignments.0.iter().cloned());
        self.nested.extend(assignments.1.iter().cloned());
    }

    fn assign(&mut self, name: &Ident) {
        if !self.direct.contains(name) {
            self.direct.insert(name.clone());
        }
    }

    fn bind(&mut self, name: &Ident) {
        self.bound.push(name.clone());
    }

    fn use_name(&mut self, name: &Ident) {
        if !self.collect_names {
            return;
        }
        // The whole-source parser shares one text per name, so the pointer
        // test settles most lookups.
        if !self
            .bound
            .iter()
            .rev()
            .any(|bound| std::sync::Arc::ptr_eq(bound, name) || **bound == **name)
            && self.seen.insert(name.clone())
        {
            self.names.push(name.clone());
        }
    }
}

/// The names one function body assigns itself, and the names functions
/// nested in it assign.
type BodyAssignments = Rc<(BTreeSet<Ident>, Rc<BTreeSet<Ident>>)>;

mod emit;
mod expressions;
mod functions;
mod loops;
mod scopes;
mod statements;

use crate::compiler::statements::statement_span;

#[cfg(test)]
mod tests;
