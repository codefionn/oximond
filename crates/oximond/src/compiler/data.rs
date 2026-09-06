//! A direct path for chunks that return data.
//!
//! Luanti stores data as Lua source and reads it back with `loadstring`:
//! `core.deserialize` and the schematic, jigsaw and settings readers load
//! chunks such as `return {name = "mcl_core:stone", prob = 254, {1, 2}}`.
//! Mineclonia's start loads 1905 of them, 6.7 MB, and building a syntax
//! tree for each only to lower it again cost more than lowering itself.
//!
//! `core.serialize` writes a string that occurs more than once as a
//! reference: `local _={};_[1]="mcl_core:stone";return {{name=_[1]}, ...}`.
//! Half of those bytes are in that form.
//!
//! This path lowers such chunks straight from the parser's tokens. A table
//! constructor is lowered here, field by field, with the compiler's own
//! register allocation, constant pool and instructions, in the order
//! [`Compiler::table_constructor`] uses. Anything else, a statement before
//! the `return` or a value that is no constructor (a number or string,
//! `-1/0`, `_[1]`), becomes a small tree that the compiler lowers as usual.
//! The chunk is therefore the one the compiler makes: the tests hold every
//! chunk this path builds against it.
//!
//! Like the whole-source parser it never reports errors. Anything else (a
//! global name, a call, another operator, another statement, a constructor
//! nested deeper than [`MAX_DEPTH`]) gives up, and the caller compiles the
//! source the usual way, which also reports any error.

use std::collections::BTreeSet;
use std::rc::Rc;
use std::sync::Arc;

use crate::bytecode::{Chunk, Constant, Instruction};
use crate::compat::CompatibilityProfile;
use crate::compiler::{
    AnalysisCache, CompileError, Compiler, CompilerConfig, Register, finish_main, new_main_compiler,
};
use crate::string::LuaString;
use crate::syntax::{
    self, BinaryOp, Expr, LocalAttribute, Name, Span, Statement, TokenKind, Tokens, UnaryOp,
};

/// The deepest nesting of constructors and unary minus this path follows.
const MAX_DEPTH: usize = 32;

/// The most constants reserved up front, about 3 MB of pool: enough for the
/// largest data Mineclonia loads (a 1.2 MB schematic). A larger source grows
/// the pool as it goes.
const MAX_RESERVED_CONSTANTS: usize = 1 << 16;

/// This path does not handle the source.
struct GiveUp;

impl From<CompileError> for GiveUp {
    fn from(_: CompileError) -> Self {
        Self
    }
}

type Lower<T> = Result<T, GiveUp>;

/// The chunk `source` compiles to, or `None` when this path does not
/// handle it.
pub(super) fn compile(
    source_name: &LuaString,
    source: &[u8],
    config: CompilerConfig,
) -> Option<Chunk> {
    // Most code starts otherwise; skip it without lexing.
    let first = source
        .iter()
        .position(|byte| !matches!(*byte, b' ' | b'\t' | b'\r' | b'\n'))?;
    if !source[first..].starts_with(b"return") && !source[first..].starts_with(b"local _") {
        return None;
    }
    let profile = config.compatibility_profile();
    let tokens = Tokens::new(source, syntax::ParserConfig::new(profile))?;
    let mut compiler = new_main_compiler(
        source_name.clone(),
        Rc::new(syntax::line_starts(source)),
        config,
        false,
        BTreeSet::new(),
        AnalysisCache::default(),
    );
    compiler.push_scope();
    compiler.bind_implicit_environment().ok()?;
    let mut data = Data {
        tokens,
        compiler,
        depth: 0,
        locals: Vec::new(),
        // Data holds a distinct number every few dozen bytes (list indices,
        // coordinates); sizing the pool once saves growing it many times.
        constants_to_reserve: (source.len() / 32).min(MAX_RESERVED_CONSTANTS),
    };
    loop {
        match data.kind() {
            TokenKind::Return => break,
            // An empty statement; the token rules reject it where the
            // profile has none.
            TokenKind::Semi => data.tokens.advance()?,
            TokenKind::Local => data.local().ok()?,
            TokenKind::Identifier(_) => data.assignment().ok()?,
            _ => return None,
        }
    }
    let span = data.return_statement().ok()?;
    // As the whole-source parser: one `;` may end the statement, and LuaJIT
    // takes any number of them after a `return`.
    let luajit = profile == CompatibilityProfile::LuaJit21;
    let mut semicolons = 0;
    while matches!(data.tokens.kind(), TokenKind::Semi) {
        semicolons += 1;
        if semicolons > 1 && !luajit {
            return None;
        }
        data.tokens.advance()?;
    }
    if !matches!(data.tokens.kind(), TokenKind::Eof) {
        return None;
    }
    finish_main(data.compiler, Some(span)).ok()
}

struct Data<'a> {
    tokens: Tokens<'a>,
    compiler: Compiler,
    depth: usize,
    /// The names of the locals declared so far. Any other name is a global,
    /// which this path leaves to the compiler.
    locals: Vec<Arc<str>>,
    /// The constant pool's room to reserve at the first table, where the
    /// source starts to look like data; zero once reserved.
    constants_to_reserve: usize,
}

impl Data<'_> {
    fn kind(&self) -> &TokenKind {
        self.tokens.kind()
    }

    fn advance(&mut self) -> Lower<()> {
        self.tokens.advance().ok_or(GiveUp)
    }

    fn expect(&mut self, kind: &TokenKind) -> Lower<()> {
        if core::mem::discriminant(self.kind()) != core::mem::discriminant(kind) {
            return Err(GiveUp);
        }
        self.advance()
    }

    fn enter(&mut self) -> Lower<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(GiveUp);
        }
        Ok(())
    }

    /// `local <name> = <value>`, lowered by the compiler.
    fn local(&mut self) -> Lower<()> {
        let start = self.tokens.span();
        self.advance()?;
        let span = self.tokens.span();
        if !matches!(self.kind(), TokenKind::Identifier(_)) {
            return Err(GiveUp);
        }
        let text = self.tokens.text(span).ok_or(GiveUp)?;
        self.advance()?;
        self.expect(&TokenKind::Assign)?;
        let value = self.statement_value()?;
        let statement = Statement::Local {
            span: start.join(value.span()),
            names: vec![Name {
                text: Arc::clone(&text),
                span,
            }],
            attributes: vec![LocalAttribute::None],
            values: vec![value],
        };
        self.compiler.statement(&statement, false)?;
        self.locals.push(text);
        Ok(())
    }

    /// `<local>[<key>]... = <value>`, lowered by the compiler.
    fn assignment(&mut self) -> Lower<()> {
        let target = self.literal()?;
        if !target.is_assignment_target() {
            return Err(GiveUp);
        }
        self.expect(&TokenKind::Assign)?;
        let value = self.statement_value()?;
        let statement = Statement::Assign {
            span: target.span().join(value.span()),
            targets: vec![target],
            values: vec![value],
        };
        self.compiler.statement(&statement, false)?;
        Ok(())
    }

    /// The value of a statement before the `return`: an empty constructor,
    /// or a value that is no constructor.
    fn statement_value(&mut self) -> Lower<Expr> {
        if !matches!(self.kind(), TokenKind::LBrace) {
            return self.quotient();
        }
        let open = self.tokens.span();
        self.advance()?;
        let close = self.tokens.span();
        self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Table {
            fields: Vec::new(),
            span: open.join(close),
        })
    }

    /// `return <value>`, lowered as `Compiler::statement` lowers it, and
    /// its span.
    fn return_statement(&mut self) -> Lower<Span> {
        let start = self.tokens.span();
        self.advance()?;
        let first_pc = self.compiler.code.len();
        let (value, value_span) = self.value()?;
        let span = start.join(value_span);
        self.compiler.close_all_scopes();
        self.compiler.code.push(Instruction::return_(value));
        let line = self.compiler.line_for(span);
        for pc in first_pc..self.compiler.code.len() {
            self.compiler.instruction_lines.insert_if_absent(pc, line);
        }
        Ok(span)
    }

    /// One value in a fresh or existing register, and its span.
    fn value(&mut self) -> Lower<(Register, Span)> {
        if matches!(self.kind(), TokenKind::LBrace) {
            return self.table();
        }
        let expression = self.quotient()?;
        let register = self.compiler.expression(&expression, 1)?;
        Ok((register, expression.span()))
    }

    /// Operands joined by `/`, which the serializer writes for infinities
    /// and NaN (`1/0`, `-1/0`, `0/0`), as the parser groups them.
    fn quotient(&mut self) -> Lower<Expr> {
        let mut left = self.unary()?;
        while matches!(self.kind(), TokenKind::Slash) {
            let operator_span = self.tokens.span();
            self.advance()?;
            let right = self.unary()?;
            left = Expr::Binary {
                op: BinaryOp::Divide,
                operator_span,
                span: left.span().join(right.span()),
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn unary(&mut self) -> Lower<Expr> {
        if !matches!(self.kind(), TokenKind::Minus) {
            return self.literal();
        }
        let span = self.tokens.span();
        self.enter()?;
        self.advance()?;
        let expression = self.unary()?;
        self.depth -= 1;
        Ok(Expr::Unary {
            op: UnaryOp::Negate,
            span: span.join(expression.span()),
            expression: Box::new(expression),
        })
    }

    fn literal(&mut self) -> Lower<Expr> {
        let span = self.tokens.span();
        let literal = match *self.kind() {
            TokenKind::Nil => Expr::Nil(span),
            TokenKind::True => Expr::Boolean(true, span),
            TokenKind::False => Expr::Boolean(false, span),
            TokenKind::Number(value) => Expr::Number(value, span),
            TokenKind::Integer(value) => Expr::Integer(value, span),
            TokenKind::String(_) => Expr::String(self.tokens.take_string(), span),
            TokenKind::Identifier(_) => return self.local_value(span),
            _ => return Err(GiveUp),
        };
        self.advance()?;
        Ok(literal)
    }

    /// A local and the fields read from it, `_[1]` or `_.name`.
    fn local_value(&mut self, span: Span) -> Lower<Expr> {
        let text = self.tokens.text(span).ok_or(GiveUp)?;
        if !self.locals.contains(&text) {
            return Err(GiveUp);
        }
        self.advance()?;
        let mut value = Expr::Name(Name { text, span });
        loop {
            value = match self.kind() {
                TokenKind::LBracket => {
                    self.enter()?;
                    self.advance()?;
                    let key = self.quotient()?;
                    let close = self.tokens.span();
                    self.expect(&TokenKind::RBracket)?;
                    self.depth -= 1;
                    Expr::Index {
                        span: value.span().join(close),
                        table: Box::new(value),
                        key: Box::new(key),
                    }
                }
                TokenKind::Dot => {
                    self.advance()?;
                    let field = self.tokens.span();
                    if !matches!(self.kind(), TokenKind::Identifier(_)) {
                        return Err(GiveUp);
                    }
                    let name = self.tokens.text(field).ok_or(GiveUp)?;
                    self.advance()?;
                    Expr::Index {
                        span: value.span().join(field),
                        table: Box::new(value),
                        key: Box::new(Expr::String(LuaString::from(&*name), field)),
                    }
                }
                // A call or method call is code.
                TokenKind::LParen | TokenKind::LBrace | TokenKind::String(_) | TokenKind::Colon => {
                    return Err(GiveUp);
                }
                _ => return Ok(value),
            };
        }
    }

    /// A table constructor, lowered as `Compiler::table_constructor` lowers
    /// it without compact registers. The size hints count the fields, so
    /// the `NewTable` is patched once they are known.
    fn table(&mut self) -> Lower<(Register, Span)> {
        let open = self.tokens.span();
        self.enter()?;
        self.advance()?;
        let reserve = std::mem::take(&mut self.constants_to_reserve);
        self.compiler.constant_indices.reserve(reserve);
        let destination = self.compiler.fresh(open)?;
        let new_table = self.compiler.code.len();
        self.compiler
            .code
            .push(Instruction::new_table(destination, 0, 0));
        let reservation = destination.checked_add(1).ok_or(GiveUp)?;
        self.compiler.reserved_register_floors.push(reservation);
        let mut list_index = 1_u32;
        let mut array = 0_usize;
        let mut hash = 0_usize;
        while !matches!(self.kind(), TokenKind::RBrace) {
            let span = self.tokens.span();
            let named = matches!(self.kind(), TokenKind::Identifier(_))
                && matches!(self.tokens.peek(), Some(TokenKind::Assign));
            if named {
                let name = self.tokens.text(span).ok_or(GiveUp)?;
                self.advance()?;
                self.advance()?;
                hash += 1;
                if let Some(field) = self.compiler.field_name_index_str(&name, span)? {
                    let (value, _) = self.value()?;
                    self.compiler
                        .code
                        .push(Instruction::set_field(destination, field, value));
                } else {
                    let key = self.compiler.string_constant_register(&name, span)?;
                    let (value, _) = self.value()?;
                    self.compiler
                        .code
                        .push(Instruction::set_table(destination, key, value));
                }
            } else if matches!(self.kind(), TokenKind::LBracket) {
                self.advance()?;
                hash += 1;
                if matches!(self.kind(), TokenKind::LBrace) {
                    return Err(GiveUp);
                }
                let (key, _) = self.value()?;
                self.expect(&TokenKind::RBracket)?;
                self.expect(&TokenKind::Assign)?;
                let (value, _) = self.value()?;
                self.compiler
                    .code
                    .push(Instruction::set_table(destination, key, value));
            } else {
                array += 1;
                let key = self
                    .compiler
                    .constant(Constant::Number(f64::from(list_index)), span)?;
                list_index = list_index.checked_add(1).ok_or(GiveUp)?;
                let (value, _) = self.value()?;
                self.compiler
                    .code
                    .push(Instruction::set_table(destination, key, value));
            }
            self.compiler.reclaim_temporaries();
            match self.kind() {
                TokenKind::Comma | TokenKind::Semi => self.advance()?,
                TokenKind::RBrace => {}
                _ => return Err(GiveUp),
            }
        }
        let close = self.tokens.span();
        self.advance()?;
        self.compiler.code[new_table] = Instruction::new_table(
            destination,
            u8::try_from(array).unwrap_or(u8::MAX),
            u8::try_from(hash).unwrap_or(u8::MAX),
        );
        self.compiler.reserved_register_floors.pop();
        self.compiler.next_register = self.compiler.next_register.max(reservation);
        self.depth -= 1;
        Ok((destination, open.join(close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CompilerLoader;
    use std::fmt::Write as _;

    const PROFILES: [CompatibilityProfile; 6] = [
        CompatibilityProfile::LuaJit21,
        CompatibilityProfile::Lua51,
        CompatibilityProfile::Lua52,
        CompatibilityProfile::Lua53,
        CompatibilityProfile::Lua54,
        CompatibilityProfile::Lua55,
    ];

    /// The chunk the compiler builds as a reader's load does without this
    /// path: through the whole-source parser, and through the streaming
    /// session for what that parser leaves to it, such as deep nesting.
    fn compiled(profile: CompatibilityProfile, source: &[u8]) -> Option<Chunk> {
        use crate::bytecode::load::{ChunkLoader as _, LoadInput, LoadPoll};
        let name = LuaString::from("=(load)");
        let loader = CompilerLoader::with_profile(profile);
        loader.compile_tree(&name, source).or_else(|| {
            let session = loader.begin(name);
            let poll = match session.resume(LoadInput::Bytes(LuaString::from(source))) {
                LoadPoll::NeedInput(next) => next.resume(LoadInput::End),
                poll => poll,
            };
            match poll {
                LoadPoll::Loaded(chunk) => Some(chunk),
                _ => None,
            }
        })
    }

    /// The chunk this path builds, checked against the compiler's: equal
    /// when both compile, and none when the compiler does not.
    fn direct(profile: CompatibilityProfile, source: &[u8]) -> Option<Chunk> {
        let name = LuaString::from("=(load)");
        let loader = CompilerLoader::with_profile(profile);
        let direct = compile(&name, source, loader.config);
        let compiled = compiled(profile, source);
        if let Some(direct) = &direct {
            let Some(compiled) = compiled else {
                panic!(
                    "{profile:?}: the compiler rejects what the direct path takes: {}",
                    String::from_utf8_lossy(source)
                );
            };
            assert_eq!(
                *direct,
                compiled,
                "{profile:?}: {}",
                String::from_utf8_lossy(source)
            );
        }
        direct
    }

    fn takes(source: &str) {
        for profile in PROFILES {
            assert!(
                direct(profile, source.as_bytes()).is_some(),
                "{profile:?} gave up on {source}"
            );
        }
    }

    fn gives_up(source: &str) {
        for profile in PROFILES {
            assert!(
                direct(profile, source.as_bytes()).is_none(),
                "{profile:?} took {source}"
            );
        }
    }

    #[test]
    fn data_chunks_compile_as_the_compiler_compiles_them() {
        takes("return {}");
        takes("return {1, 2, 3}");
        takes("return {name = \"mcl_core:stone\", prob = 254, param2 = 0}");
        takes("return {1, \"a\\n\", x = 2.5, [3] = true, {y = 1, z = 2}, -1/0, 0/0, 1/0}");
        takes("return {[\"key with space\"] = {}, [1.5] = false, [-2] = nil; 7;}");
        takes("return {a = 1, a = 2, [1] = 3, 4, [2] = 5, 6}");
        takes("return {-0, - -3, 1e308, 1e309, -1e309}");
        takes(
            "local _={};_[1]=\"mcl_core:stone\";_[2]=\"x\";\
             return {{name=_[1]},{name=_[1],[_[2]]=_[2]},_[1]}",
        );
        takes("local _={};_[1]={};_[2]={};_[1].self=_[1];_[1][_[2]]=_[2];_[2][1]=-1/0;return _[1]");
        takes("local _ = {}\n_[1] = 'a'\n_[2] = _[1]\nreturn {_[2], _.x}");
        takes("local _={};return {}");
        takes("return 5");
        takes("return -5");
        takes("return 'text'");
        takes("return nil;");
        takes("return {size = {x = 1, y = 2, z = 3}}");
        takes("  \n\treturn {\n  1,\n  \"two\\\n lines\",\n  {x = 1}\n}\n");
        takes("return {1, 2} -- trailing comment");
        takes("return { --[[ long comment ]] 1, [==[long string]==], 0x10, 3e2, .5 }");
        takes("return {\"\\65\\066\\\\\\r\", '\\'', \"\\\"\"}");
    }

    #[test]
    fn code_is_left_to_the_compiler() {
        gives_up("local _={};_[1]=f();return {_[1]}");
        gives_up("local x={};return {y[1]}");
        gives_up("local _={};return {_[1]()}");
        gives_up("local _={};return {_:m()}");
        gives_up("local _ <const> = {};return {}");
        gives_up("local a, b = {}, {};return {}");
        gives_up("return {x = y}");
        gives_up("return {f()}");
        gives_up("return {...}");
        gives_up("return {1 + 2}");
        gives_up("return {{} / 2}");
        gives_up("return {[{}] = 1}");
        gives_up("return 1, 2");
        gives_up("return {1}, 2");
        gives_up("return");
        gives_up("x = 1 return {}");
        gives_up("return {1} x = 1");
        gives_up("return {1,,2}");
        gives_up("return {1");
        gives_up("return {\"unfinished}");
        gives_up("return #{}");
        gives_up("return {loadstring(\"\\27Lua\")}");
        gives_up(&format!("return {}1{}", "{".repeat(40), "}".repeat(40)));
    }

    #[test]
    fn deep_nesting_compiles_as_the_compiler_compiles_it() {
        for depth in 20..=MAX_DEPTH {
            let open = "{".repeat(depth);
            let close = "}".repeat(depth);
            takes(&format!("return {open}1{close}"));
            takes(&format!("return {open}x = 1, 2, 'a'{close}"));
            // An index or a minus counts as a level too.
            let (open, close) = ("{".repeat(depth - 1), "}".repeat(depth - 1));
            takes(&format!(
                "return {{{{1}}, {open}2{close}, x = {{}}, {open}3{close}}}"
            ));
            takes(&format!(
                "local _={{}};_[1]='a';return {open}_[1], -1/0{close}"
            ));
        }
        let (open, close) = ("{".repeat(MAX_DEPTH), "}".repeat(MAX_DEPTH));
        gives_up(&format!("return {{{open}1{close}}}"));
        gives_up(&format!("local _={{}};_[1]='a';return {open}_[1]{close}"));
        gives_up(&format!("return {open}-1{close}"));
    }

    #[test]
    fn operands_across_lines_keep_the_compilers_line_numbers() {
        takes("return {1\n/\n0, -\n1\n/\n0, -\n-\n2}");
        takes("return 1\n/\n0");
        takes("local _={}\n_[\n1\n]\n=\n'a'\nreturn {_[\n1\n], [\n_[\n1\n]\n]\n=\n_[\n1\n]}");
        takes("local _={};_[1]={};_[\n1\n]\n.\nself\n=\n_[1];return _[\n1\n]");
        takes("return {\r\n1\r\n,\r\n-\r\n1\r\n/\r\n0, x\r\n=\r\n'a'\r\n}");
        takes(
            "local _={}\r\n_[1]\r\n=\r\n{}\r\n_[\r\n1\r\n][\r\n2\r\n]\r\n=\r\n0\r\nreturn {_[\r\n1]}",
        );
        takes("return {'a\\\r\nb', \"c\\\n\rd\"}");
    }

    #[test]
    fn a_bare_local_is_returned_as_the_compiler_returns_it() {
        takes("local _={};return _");
        takes("local _={};_[1]=1;_[2]={};_[2][1]=_[1];return _");
        takes("local _ = {}\nreturn _\n");
    }

    #[test]
    fn goto_is_a_field_name_in_luajit_only() {
        let source = b"return {goto = 1}";
        assert!(direct(CompatibilityProfile::LuaJit21, source).is_some());
        assert!(direct(CompatibilityProfile::Lua54, source).is_none());
    }

    #[test]
    fn newer_escapes_are_left_to_profiles_that_reject_them() {
        let source = b"return {\"\\x43\\u{44}\\z   E\"}";
        assert!(direct(CompatibilityProfile::LuaJit21, source).is_some());
        assert!(direct(CompatibilityProfile::Lua54, source).is_some());
        assert!(direct(CompatibilityProfile::Lua51, source).is_none());
    }

    #[test]
    fn luajit_takes_semicolons_after_return() {
        let source = b"return {1};;";
        assert!(direct(CompatibilityProfile::LuaJit21, source).is_some());
        assert!(direct(CompatibilityProfile::Lua51, source).is_none());
    }

    #[test]
    fn many_distinct_names_and_constants_use_the_wide_forms() {
        let mut source = String::from("return {");
        for index in 0..700 {
            let _ = write!(source, "k{index} = {index}.25, \"s{index}\", ");
        }
        source.push('}');
        takes(&source);
    }

    /// A small deterministic generator (xorshift64*).
    /// The state, and how many `_[n]` references the chunk declared.
    struct Random(u64, u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }

        fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
            items[usize::try_from(self.below(items.len() as u64)).expect("small index")]
        }
    }

    fn space(random: &mut Random, out: &mut String) {
        out.push_str(random.pick(&["", "", "", " ", "\n", "\r\n", "\t", " --c\n"]));
    }

    fn number(random: &mut Random) -> String {
        match random.below(10) {
            0 => format!("{}", random.below(10)),
            1 => format!("{}", random.below(1 << 40)),
            2 => format!("{:.17e}", f64::from_bits(random.next() >> 2)),
            3 => format!("{}.{}", random.below(1000), random.below(1000)),
            4 => format!("0x{:x}", random.below(1 << 20)),
            5 => random
                .pick(&["1/0", "-1/0", "0/0", "-0", "1e309", "- -2"])
                .to_owned(),
            6 => format!("-{}", random.below(300)),
            7 => format!("{}e{}", random.below(100), random.below(40)),
            8 => "9007199254740993".to_owned(),
            _ => format!("{}", random.below(255)),
        }
    }

    fn string(random: &mut Random) -> String {
        let mut text = String::from("\"");
        for _ in 0..random.below(8) {
            text.push_str(random.pick(&[
                "a",
                "mcl_core:stone",
                "\\n",
                "\\\\",
                "\\\"",
                "\\0",
                "\\255",
                "\\\n",
                "'",
                " ",
                "\\r",
                "\\t",
                "x",
            ]));
        }
        text.push('"');
        text
    }

    fn value(random: &mut Random, depth: usize, out: &mut String) {
        match random.below(if depth < 4 { 8 } else { 6 }) {
            0 | 1 => out.push_str(&number(random)),
            2 | 3 => out.push_str(&string(random)),
            4 => out.push_str(random.pick(&["true", "false", "nil"])),
            5 if random.1 > 0 => {
                let refs = random.1;
                let _ = write!(out, "_[{}]", 1 + random.below(refs));
            }
            5 => {
                let _ = write!(out, "'k{}'", random.below(4));
            }
            _ => table(random, depth + 1, out),
        }
    }

    fn table(random: &mut Random, depth: usize, out: &mut String) {
        out.push('{');
        let bound = if random.below(8) == 0 { 300 } else { 6 };
        let fields = random.below(bound);
        for index in 0..fields {
            space(random, out);
            match random.below(4) {
                0 => {
                    out.push_str(random.pick(&["name", "prob", "param2", "x", "y", "goto"]));
                    space(random, out);
                    out.push('=');
                }
                1 => {
                    out.push('[');
                    if random.below(2) == 0 {
                        out.push_str(&number(random));
                    } else {
                        out.push_str(&string(random));
                    }
                    out.push_str("]=");
                }
                _ => {}
            }
            space(random, out);
            value(random, depth, out);
            if index + 1 < fields || random.below(3) == 0 {
                out.push_str(random.pick(&[",", ",", ";"]));
            }
        }
        space(random, out);
        out.push('}');
    }

    #[test]
    fn generated_data_chunks_compile_as_the_compiler_compiles_them() {
        let mut random = Random(0x9E37_79B9_7F4A_7C15, 0);
        let mut taken = 0;
        let rounds = if cfg!(debug_assertions) { 300 } else { 3000 };
        for round in 0..rounds {
            // Every third chunk in the serializer's reference form.
            random.1 = 0;
            let mut source = String::new();
            if round % 3 == 0 {
                source.push_str("local _={};");
                random.1 = random.below(4);
                for index in 1..=random.1 {
                    let text = string(&mut random);
                    let _ = write!(source, "_[{index}]={text};");
                }
            }
            source.push_str("return ");
            if round % 5 == 0 {
                value(&mut random, 0, &mut source);
            } else {
                table(&mut random, 0, &mut source);
            }
            if random.below(4) == 0 {
                source.push(';');
            }
            let profile = PROFILES[round % PROFILES.len()];
            // Every generated source the compiler takes is data, which
            // the direct path takes too.
            let compiles = compiled(profile, source.as_bytes()).is_some();
            let took = direct(profile, source.as_bytes()).is_some();
            assert_eq!(took, compiles, "{profile:?}: {source}");
            taken += usize::from(took);
            // Every cut of the source is either taken alike or left alone.
            let cut = usize::try_from(random.below(source.len() as u64)).expect("small index");
            if source.is_char_boundary(cut) {
                let _ = direct(profile, &source.as_bytes()[..cut]);
            }
        }
        // Only `goto` keys, a keyword since Lua 5.2, make sources invalid.
        assert!(taken > rounds * 3 / 4, "{taken} of {rounds}");
    }
}
