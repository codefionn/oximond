#![forbid(unsafe_code)]

//! Lua 5.1 byte-pattern matching.
//!
//! This is deliberately a small, standalone engine. It accepts arbitrary
//! bytes, does not borrow a host regex implementation, and applies fixed
//! limits to the two resources Lua patterns can otherwise consume without
//! bound: recursive matching depth and backtracking steps.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::rc::Rc;

const MAX_PATTERN_BYTES: usize = 4_096;
/// The longest pattern [`Pattern::shared`] keeps.
pub const CACHED_PATTERN_BYTES: usize = 128;
/// How many patterns of each kind [`Pattern::shared`] keeps.
const CACHED_PATTERNS: usize = 1_024;
const MAX_CAPTURES: usize = 32;
const MAX_DEPTH: usize = 512;
const MAX_STEPS: usize = 2_000_000;

/// FNV-1a, which hashes a short pattern faster than the default hasher.
#[derive(Default)]
struct Fnv(u64);

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = if self.0 == 0 {
            0xcbf2_9ce4_8422_2325
        } else {
            self.0
        };
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        self.0 = hash;
    }
}

type PatternMap = HashMap<Box<[u8]>, Rc<Pattern>, BuildHasherDefault<Fnv>>;

/// How many patterns of each kind [`Pattern::shared`] finds by the address of
/// the bytes it was given, without hashing them.
const RECENT_SLOTS: usize = 4096;

/// The patterns of [`Pattern::shared`]: one set for the patterns of the other
/// functions, one for `string.gmatch`'s.
struct SharedPatterns {
    maps: [PatternMap; 2],
}

thread_local! {
    static CACHE: RefCell<SharedPatterns> = RefCell::new(SharedPatterns {
        maps: [PatternMap::default(), PatternMap::default()],
    });
    /// The patterns [`Pattern::shared`] found last, by the address of the
    /// bytes that named them (see [`RECENT_SLOTS`]): the kinds' slots one
    /// after the other. The slots live on the heap from the first shared
    /// pattern on. Held in the thread local itself, they put 64 KiB into the
    /// initial TLS image, which the loader copies into every thread at start
    /// whether it matches a pattern or not: 16 page faults on each start.
    static RECENT: OnceCell<Box<RecentSlots>> = const { OnceCell::new() };
}

type RecentSlots = [Cell<Option<Rc<Pattern>>>; 2 * RECENT_SLOTS];

fn empty_recent_slots() -> Box<RecentSlots> {
    let slots: Box<[_]> = (0..2 * RECENT_SLOTS).map(|_| Cell::new(None)).collect();
    match slots.try_into() {
        Ok(slots) => slots,
        Err(_) => unreachable!("the iterator yields 2 * RECENT_SLOTS slots"),
    }
}

/// One completed capture. Position captures have an empty range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capture {
    start: usize,
    end: usize,
    position: bool,
}

impl Capture {
    #[must_use]
    pub const fn start(self) -> usize {
        self.start
    }
    #[must_use]
    pub const fn end(self) -> usize {
        self.end
    }
    #[must_use]
    pub const fn is_position(self) -> bool {
        self.position
    }
}

/// A successful byte-range match.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Match {
    start: usize,
    end: usize,
    captures: Vec<Capture>,
}

impl Match {
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }
    #[must_use]
    pub fn captures(&self) -> &[Capture] {
        &self.captures
    }
}

/// A `string.gsub` replacement string parsed once. It covers only the escapes
/// every profile accepts (`%%` and `%0` to `%9`), so a template with another
/// `%` has no parsed form and is expanded by the caller's profile rules.
#[derive(Clone, Debug)]
pub struct Template {
    bytes: Vec<u8>,
    parts: Vec<TemplatePart>,
}

#[derive(Clone, Debug)]
enum TemplatePart {
    Text(Vec<u8>),
    /// `%0`.
    Whole,
    /// `%1` to `%9`, as the capture index from zero.
    Capture(usize),
}

impl Template {
    /// The parsed form of `bytes`, or `None` for a `%` before another
    /// character or at the end.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let mut parts = Vec::new();
        let mut text = Vec::new();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] != b'%' {
                text.push(bytes[index]);
                index += 1;
                continue;
            }
            let part = match bytes.get(index + 1)? {
                b'%' => {
                    text.push(b'%');
                    index += 2;
                    continue;
                }
                b'0' => TemplatePart::Whole,
                digit @ b'1'..=b'9' => TemplatePart::Capture(usize::from(digit - b'1')),
                _ => return None,
            };
            if !text.is_empty() {
                parts.push(TemplatePart::Text(std::mem::take(&mut text)));
            }
            parts.push(part);
            index += 2;
        }
        if !text.is_empty() {
            parts.push(TemplatePart::Text(text));
        }
        Some(Self {
            bytes: bytes.to_vec(),
            parts,
        })
    }

    /// The template's text.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Whether the template has no `%`.
    #[must_use]
    pub fn is_plain(&self) -> bool {
        !self.bytes.contains(&b'%')
    }

    /// Appends the template for the match `found` of `source`. `false` when
    /// the template names a capture the match lacks.
    #[must_use]
    pub fn expand(&self, output: &mut Vec<u8>, source: &[u8], found: &Match) -> bool {
        for part in &self.parts {
            match part {
                TemplatePart::Text(text) => output.extend_from_slice(text),
                TemplatePart::Whole => output.extend_from_slice(&source[found.start..found.end]),
                TemplatePart::Capture(0) if found.captures.is_empty() => {
                    output.extend_from_slice(&source[found.start..found.end]);
                }
                TemplatePart::Capture(index) => {
                    let Some(capture) = found.captures.get(*index) else {
                        return false;
                    };
                    if capture.is_position() {
                        output.extend_from_slice((capture.start() + 1).to_string().as_bytes());
                    } else {
                        output.extend_from_slice(&source[capture.start()..capture.end()]);
                    }
                }
            }
        }
        true
    }
}

/// What a trace prepares for a `string.gsub` with a constant pattern: the
/// pattern and, when the replacement is a constant string with only
/// universal escapes, its parsed template.
#[derive(Debug)]
pub struct GsubPlan {
    pub pattern: Pattern,
    pub template: Option<Template>,
}

/// The shape of a single-run pattern (see [`Pattern::run_class`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunShape {
    /// Whether the pattern starts with `^`.
    pub anchored: bool,
    /// Whether the run is the pattern's one capture (`(%d+)`).
    pub captured: bool,
}

/// A match as the matcher holds it, borrowed: [`Match`] without copying the
/// captures out.
pub struct MatchRef<'a> {
    start: usize,
    end: usize,
    slots: &'a [Slot],
}

impl MatchRef<'_> {
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }
    /// How many captures the match has.
    #[must_use]
    pub const fn capture_count(&self) -> usize {
        self.slots.len()
    }
    /// The capture at `index`, which is below [`Self::capture_count`].
    #[must_use]
    pub fn capture(&self, index: usize) -> Capture {
        capture_of(self.slots[index])
    }
}

/// A set of subject bytes as a 256-bit table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ByteSet([u64; 4]);

impl ByteSet {
    const EMPTY: Self = Self([0; 4]);

    const fn with(mut self, byte: u8) -> Self {
        self.0[(byte >> 6) as usize] |= 1 << (byte & 63);
        self
    }

    fn insert(&mut self, byte: u8) {
        *self = self.with(byte);
    }

    fn union(&mut self, other: Self) {
        for (word, other) in self.0.iter_mut().zip(other.0) {
            *word |= other;
        }
    }

    fn negated(self) -> Self {
        Self(self.0.map(|word| !word))
    }

    #[inline]
    fn contains(&self, byte: u8) -> bool {
        (self.0[usize::from(byte >> 6)] >> (byte & 63)) & 1 != 0
    }
}

/// The classes `%a %c %d %g %l %p %s %u %w %x %z`, by lowercase letter.
const CLASS_LETTERS: [u8; 11] = *b"acdglpsuwxz";

const fn class_member(letter: u8, byte: u8) -> bool {
    match letter {
        b'a' => byte.is_ascii_alphabetic(),
        b'c' => byte.is_ascii_control(),
        b'd' => byte.is_ascii_digit(),
        b'g' => byte.is_ascii_graphic(),
        b'l' => byte.is_ascii_lowercase(),
        b'p' => byte.is_ascii_punctuation(),
        b's' => byte.is_ascii_whitespace() || byte == 11,
        b'u' => byte.is_ascii_uppercase(),
        b'w' => byte.is_ascii_alphanumeric(),
        b'x' => byte.is_ascii_hexdigit(),
        _ => byte == 0,
    }
}

const CLASS_SETS: [ByteSet; 11] = {
    let mut sets = [ByteSet::EMPTY; 11];
    let mut class = 0;
    while class < sets.len() {
        let mut byte = 0_u8;
        loop {
            if class_member(CLASS_LETTERS[class], byte) {
                sets[class] = sets[class].with(byte);
            }
            if byte == u8::MAX {
                break;
            }
            byte += 1;
        }
        class += 1;
    }
    sets
};

/// The bytes `%class` matches. An uppercase class letter is the complement of
/// the lowercase one; any other byte stands for itself.
fn class_set(class: u8) -> ByteSet {
    match CLASS_LETTERS
        .iter()
        .position(|&letter| letter == class.to_ascii_lowercase())
    {
        Some(index) if class.is_ascii_uppercase() => CLASS_SETS[index].negated(),
        Some(index) => CLASS_SETS[index],
        None => ByteSet::EMPTY.with(class),
    }
}

/// The bytes of the `[set]` that starts at `start`, which passed `class_end`.
fn set_table(pattern: &[u8], start: usize) -> Result<ByteSet, PatternError> {
    let mut index = start + 1;
    let complement = pattern.get(index) == Some(&b'^');
    if complement {
        index += 1;
    }
    let mut set = ByteSet::EMPTY;
    // Lua accepts `]` as the first member of a set, including after `^`,
    // and as the start of a range there (`[]-a]`).
    let first = index;
    while let Some(&current) = pattern.get(index) {
        if current == b']' && index != first {
            return Ok(if complement { set.negated() } else { set });
        }
        if current == b'%' {
            let class = *pattern
                .get(index + 1)
                .ok_or(PatternError::MalformedPattern("missing class"))?;
            set.union(class_set(class));
            index += 2;
        } else if pattern.get(index + 1) == Some(&b'-')
            && pattern.get(index + 2).is_some_and(|next| *next != b']')
        {
            for byte in current..=pattern[index + 2] {
                set.insert(byte);
            }
            index += 3;
        } else {
            set.insert(current);
            index += 1;
        }
    }
    Err(PatternError::MalformedPattern("missing ']'"))
}

/// One subject byte test: `.`, a literal byte, or a class or set resolved to
/// a table when the pattern was compiled.
#[derive(Clone, Copy, Debug)]
enum Atom {
    Any,
    Byte(u8),
    Set(ByteSet),
}

impl Atom {
    #[inline]
    fn matches(&self, byte: u8) -> bool {
        match self {
            Self::Any => true,
            Self::Byte(expected) => byte == *expected,
            Self::Set(set) => set.contains(byte),
        }
    }

    /// Whether no byte `need` accepts is one the atom accepts.
    fn is_apart_from(&self, need: Skip) -> bool {
        match (self, need) {
            (Self::Any, _) => false,
            (Self::Byte(byte), need) => !need.matches(*byte),
            (Self::Set(set), Skip::Byte(byte)) => !set.contains(byte),
            (Self::Set(set), Skip::Set(other)) => {
                set.0.iter().zip(other.0).all(|(a, b)| a & b == 0)
            }
        }
    }

    /// How many bytes at the front of `bytes` the atom matches.
    #[inline]
    fn run_len(&self, bytes: &[u8]) -> usize {
        match self {
            Self::Any => bytes.len(),
            Self::Byte(expected) => bytes
                .iter()
                .position(|byte| byte != expected)
                .unwrap_or(bytes.len()),
            Self::Set(set) => bytes
                .iter()
                .position(|&byte| !set.contains(byte))
                .unwrap_or(bytes.len()),
        }
    }
}

/// One pre-decoded pattern item. The matcher walks these instead of the
/// pattern bytes, so it never parses class or set syntax per subject byte.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// A run of literal bytes, `len` bytes at `start` in the pattern's
    /// literal table.
    Lit {
        start: u32,
        len: u32,
    },
    One(Atom),
    /// `atom?`
    Opt(Atom),
    /// `atom*`
    Star(Atom),
    /// `atom+`
    Plus(Atom),
    /// `atom-`
    Lazy(Atom),
    /// `atom*` followed by an op that needs a byte the atom never accepts.
    /// Only the end of the run can be followed by the rest of the pattern, so
    /// backtracking inside the run is skipped.
    StarApart(Atom),
    /// `atom+`, as [`Op::StarApart`].
    PlusApart(Atom),
    /// `atom-` followed by an op that needs a byte the atom never accepts:
    /// the rest of the pattern can only follow where the atom stops
    /// accepting.
    LazyApart(Atom),
    Open,
    Position,
    Close,
    /// `(atom+)`: an [`Op::Open`], [`Op::Plus`] and [`Op::Close`] in one op.
    /// It counts the steps and the depth of the three, so the resource limits
    /// trip where they would for the three.
    CapturePlus(Atom),
    /// `(atom*)`, as [`Op::CapturePlus`].
    CaptureStar(Atom),
    /// `(atom-)`, as [`Op::CapturePlus`].
    CaptureLazy(Atom),
    Back(u8),
    Balance(u8, u8),
    Frontier(ByteSet),
    /// `$` at the end of the pattern.
    End,
}

/// The first byte any match must start with, for the scan loop to skip to.
#[derive(Clone, Copy, Debug)]
enum Skip {
    Byte(u8),
    Set(ByteSet),
}

impl Skip {
    /// The filter for exactly the bytes of `set`.
    fn of(set: ByteSet) -> Self {
        let mut bytes = (0..=u8::MAX).filter(|&byte| set.contains(byte));
        match (bytes.next(), bytes.next()) {
            (Some(only), None) => Self::Byte(only),
            _ => Self::Set(set),
        }
    }

    #[inline]
    fn matches(&self, byte: u8) -> bool {
        match self {
            Self::Byte(expected) => byte == *expected,
            Self::Set(set) => set.contains(byte),
        }
    }
}

/// A validated Lua byte pattern, compiled to [`Op`]s.
#[derive(Clone, Debug)]
pub struct Pattern {
    bytes: Vec<u8>,
    anchored: bool,
    ops: Vec<Op>,
    literals: Vec<u8>,
    skip: Option<Skip>,
    /// For a [`Skip::Set`], the set as a table by byte value (1 for a member),
    /// which the scan loop reads a byte at a time without the shifts a bit
    /// set needs.
    scan: Option<Box<[u8; 256]>>,
    /// The atom of a pattern that is a single `atom+` (`%a+`, `[%w_]+`),
    /// which matches without the matcher: a run of the atom's bytes.
    run_of: Option<Atom>,
    /// The op whose run decides where the next search may start after a failed
    /// one (see [`restart_of`]): its index, or `usize::MAX`, and whether the
    /// position the run ends at is covered too.
    restart_pc: usize,
    restart_past_end: bool,
    /// Per op, the byte the subject must have at the op's start for it to
    /// match at all, where that is known. A backtracking loop tests this
    /// before it enters the rest of the pattern.
    needs: Vec<Option<Skip>>,
    /// How many captures the pattern opens.
    captures: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PatternError {
    TooLong,
    MalformedPattern(&'static str),
    /// A back-reference `%N` names a capture that is not complete; carries `N`.
    InvalidCapture(u8),
    ResourceLimit,
}

/// What stops a match: [`PatternError`] without its payload text, so that the
/// matcher's result is two words and comes back in registers.
#[derive(Clone, Copy)]
enum Fault {
    ResourceLimit,
    TooManyCaptures,
    InvalidClose,
    InvalidCapture(u8),
}

impl From<Fault> for PatternError {
    #[cold]
    fn from(fault: Fault) -> Self {
        match fault {
            Fault::ResourceLimit => Self::ResourceLimit,
            Fault::TooManyCaptures => Self::MalformedPattern("too many captures"),
            Fault::InvalidClose => Self::MalformedPattern("invalid ')'"),
            Fault::InvalidCapture(number) => Self::InvalidCapture(number),
        }
    }
}

impl PatternError {
    /// The message stock Lua raises for this error. `legacy` selects the
    /// wording of Lua 5.1 and `LuaJIT`, which differs for `%b` without
    /// arguments and names no capture index.
    #[must_use]
    pub fn lua_message(&self, legacy: bool) -> String {
        match self {
            Self::TooLong => "pattern is too long".to_owned(),
            Self::MalformedPattern("missing arguments to %b") if legacy => {
                "unbalanced pattern".to_owned()
            }
            Self::MalformedPattern("missing arguments to %b") => {
                "malformed pattern (missing arguments to '%b')".to_owned()
            }
            Self::MalformedPattern("missing '[' after %f") => {
                "missing '[' after '%f' in pattern".to_owned()
            }
            Self::MalformedPattern("invalid ')'") => "invalid pattern capture".to_owned(),
            Self::MalformedPattern(detail @ ("too many captures" | "unfinished capture")) => {
                (*detail).to_owned()
            }
            Self::MalformedPattern(detail) => format!("malformed pattern ({detail})"),
            Self::InvalidCapture(_) if legacy => "invalid capture index".to_owned(),
            Self::InvalidCapture(index) => format!("invalid capture index %{index}"),
            Self::ResourceLimit => "pattern too complex".to_owned(),
        }
    }
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.lua_message(false))
    }
}
impl Error for PatternError {}

impl Pattern {
    /// Validates a Lua 5.1 pattern.
    ///
    /// # Errors
    ///
    /// Returns a syntax or configured-size error for an invalid pattern.
    pub fn new(bytes: impl AsRef<[u8]>) -> Result<Self, PatternError> {
        Self::build(bytes.as_ref(), true)
    }

    /// Validates a `string.gmatch` pattern. Stock Lua gives a leading `^`
    /// no meaning there: it matches a literal caret.
    ///
    /// # Errors
    ///
    /// As [`Pattern::new`].
    pub fn new_for_gmatch(bytes: impl AsRef<[u8]>) -> Result<Self, PatternError> {
        Self::build(bytes.as_ref(), false)
    }

    fn build(bytes: &[u8], anchors: bool) -> Result<Self, PatternError> {
        let bytes = bytes.to_vec();
        if bytes.len() > MAX_PATTERN_BYTES {
            return Err(PatternError::TooLong);
        }
        validate(&bytes)?;
        let anchored = anchors && bytes.first() == Some(&b'^');
        let (ops, literals) = compile(&bytes[usize::from(anchored)..])?;
        let captures = ops
            .iter()
            .filter(|op| matches!(op, Op::Open | Op::Position))
            .count();
        let skip = start_filter(&ops, &literals);
        let ops = fuse_captures(&ops);
        let (restart_pc, restart_past_end) = restart_of(&ops).unwrap_or((usize::MAX, false));
        let run_of = match ops.as_slice() {
            [Op::Plus(atom)] => Some(*atom),
            _ => None,
        };
        let needs: Vec<_> = ops.iter().map(|op| need_of(op, &literals)).collect();
        let scan = match skip {
            Some(Skip::Set(set)) => {
                let mut table = Box::new([0_u8; 256]);
                for byte in 0..=u8::MAX {
                    table[usize::from(byte)] = u8::from(set.contains(byte));
                }
                Some(table)
            }
            _ => None,
        };
        let ops = ops
            .iter()
            .enumerate()
            .map(
                |(pc, op)| match (*op, needs.get(pc + 1).copied().flatten()) {
                    (Op::Star(atom), Some(need)) if atom.is_apart_from(need) => Op::StarApart(atom),
                    (Op::Plus(atom), Some(need)) if atom.is_apart_from(need) => Op::PlusApart(atom),
                    (Op::Lazy(atom), Some(need)) if atom.is_apart_from(need) => Op::LazyApart(atom),
                    (op, _) => op,
                },
            )
            .collect();
        Ok(Self {
            bytes,
            anchored,
            ops,
            literals,
            skip,
            scan,
            run_of,
            restart_pc,
            restart_past_end,
            needs,
            captures,
        })
    }

    /// The validated pattern for `bytes`, shared with every earlier call that
    /// named the same bytes on this thread (`gmatch` selects the pattern of
    /// `string.gmatch`, see [`Pattern::new_for_gmatch`]). A program that
    /// matches a few patterns over and over compiles each once. Patterns
    /// longer than [`CACHED_PATTERN_BYTES`] are built every time.
    ///
    /// # Errors
    ///
    /// As [`Pattern::new`].
    pub fn shared(bytes: &[u8], gmatch: bool) -> Result<Rc<Self>, PatternError> {
        if bytes.len() > CACHED_PATTERN_BYTES {
            return Self::build(bytes, !gmatch).map(Rc::new);
        }
        let kind = usize::from(gmatch);
        // The same interned string arrives at the same address again and
        // again. Look there first and compare the bytes, which is cheaper
        // than hashing them and stays correct when the address has been
        // reused for another string.
        let slot = kind * RECENT_SLOTS
            + (((bytes.as_ptr() as usize >> 4) ^ (bytes.len() << 7))
                .wrapping_mul((0x9E37_79B9_7F4A_7C15_u64 >> (u64::BITS - usize::BITS)) as usize)
                >> (usize::BITS - RECENT_SLOTS.trailing_zeros()));
        // `try_with` rather than `with`: without its panic path the access is
        // small enough that the compiler inlines it, and a hit costs no call.
        let recent = RECENT
            .try_with(|recent| {
                let cell = &recent.get()?[slot];
                let held = cell.take();
                let found = held
                    .as_ref()
                    .filter(|found| found.bytes == bytes)
                    .map(Rc::clone);
                cell.set(held);
                found
            })
            .ok()
            .flatten();
        if let Some(found) = recent {
            return Ok(found);
        }
        let pattern = CACHE.with(|cache| -> Result<Rc<Self>, PatternError> {
            let mut cache = cache.borrow_mut();
            let map = &mut cache.maps[kind];
            if let Some(found) = map.get(bytes) {
                return Ok(Rc::clone(found));
            }
            let pattern = Rc::new(Self::build(bytes, !gmatch)?);
            if map.len() >= CACHED_PATTERNS {
                // A program with that many patterns gains nothing from a
                // partial cache: start over.
                map.clear();
                RECENT.with(|recent| {
                    if let Some(recent) = recent.get() {
                        recent.iter().for_each(|cell| drop(cell.take()));
                    }
                });
            }
            map.insert(bytes.into(), Rc::clone(&pattern));
            Ok(pattern)
        })?;
        RECENT.with(|recent| {
            recent.get_or_init(empty_recent_slots)[slot].set(Some(Rc::clone(&pattern)));
        });
        Ok(pattern)
    }

    /// The offset of the first byte of `bytes` a match can start with. Only
    /// for a pattern with a start filter.
    #[inline]
    fn first_candidate(&self, skip: &Skip, bytes: &[u8]) -> Option<usize> {
        match (skip, &self.scan) {
            (Skip::Byte(expected), _) => position_of_byte(*expected, bytes),
            (Skip::Set(_), Some(table)) => position_in_table(table, bytes),
            (Skip::Set(set), None) => bytes.iter().position(|&byte| set.contains(byte)),
        }
    }

    /// [`Self::find`] for a pattern that is a single `atom+`: the first run of
    /// the atom's bytes at or after `initial`, as its start and end. The
    /// matcher would take two steps and no backtracking for it, so no limit
    /// can trip.
    #[inline]
    fn find_run(&self, atom: &Atom, subject: &[u8], initial: usize) -> Option<(usize, usize)> {
        let from = initial.min(subject.len());
        let start = match (&self.skip, self.anchored) {
            (_, true) | (None, false) => from,
            (Some(skip), false) => from + self.first_candidate(skip, &subject[from..])?,
        };
        if !subject.get(start).is_some_and(|&byte| atom.matches(byte)) {
            return None;
        }
        let after = start + 1;
        Some((start, after + atom.run_len(&subject[after..])))
    }

    /// For a pattern that is a single run of one atom (`%d+`, `[^,]+`,
    /// `^%a+`), or one captured whole (`(%d+)`): the atom as a table by byte
    /// value (1 for a member), and the run's shape. A match is the first run
    /// of members at or after the start (at the start itself when anchored),
    /// and the capture, if any, is its text.
    #[must_use]
    pub fn run_class(&self) -> Option<(Box<[u8; 256]>, RunShape)> {
        let (atom, captured) = match (self.run_of, &self.ops[..]) {
            (Some(atom), _) => (atom, false),
            (None, [Op::CapturePlus(atom)]) => (*atom, true),
            _ => return None,
        };
        let mut table = Box::new([0_u8; 256]);
        for byte in 0..=u8::MAX {
            table[usize::from(byte)] = u8::from(atom.matches(byte));
        }
        Some((
            table,
            RunShape {
                anchored: self.anchored,
                captured,
            },
        ))
    }

    /// The bytes the pattern was built from, anchor included.
    #[must_use]
    pub fn source(&self) -> &[u8] {
        &self.bytes
    }

    /// Whether the pattern starts with `^`.
    #[must_use]
    pub const fn is_anchored(&self) -> bool {
        self.anchored
    }

    /// Finds the first match at or after `initial`, measured in bytes.
    ///
    /// # Errors
    ///
    /// Returns a bounded-resource error when matching exceeds the engine's
    /// deterministic recursion or backtracking budget.
    pub fn find(&self, subject: &[u8], initial: usize) -> Result<Option<Match>, PatternError> {
        if let Some(atom) = &self.run_of {
            return Ok(self
                .find_run(atom, subject, initial)
                .map(|(start, end)| Match {
                    start,
                    end,
                    captures: Vec::new(),
                }));
        }
        let mut start = initial.min(subject.len());
        let mut matcher = Matcher {
            subject,
            pattern: self,
            captures: Slots {
                inline: [EMPTY_SLOT; INLINE_CAPTURES],
                spill: if self.captures > INLINE_CAPTURES {
                    vec![EMPTY_SLOT; MAX_CAPTURES]
                } else {
                    Vec::new()
                },
            },
            open: 0,
            steps: 0,
            restart: 0,
        };
        loop {
            if let Some(skip) = &self.skip {
                if self.anchored {
                    if !subject.get(start).is_some_and(|&byte| skip.matches(byte)) {
                        return Ok(None);
                    }
                } else if let Some(offset) = self.first_candidate(skip, &subject[start..]) {
                    start += offset;
                } else {
                    return Ok(None);
                }
            }
            matcher.open = 0;
            matcher.steps = 0;
            matcher.restart = 0;
            if let Some(end) = matcher.run(0, start, 0).map_err(PatternError::from)? {
                return Ok(Some(Match {
                    start,
                    end,
                    captures: matcher.captures.all()[..matcher.open]
                        .iter()
                        .map(|slot| capture_of(*slot))
                        .collect(),
                }));
            }
            if self.anchored || start >= subject.len() {
                return Ok(None);
            }
            // The starts inside the run the failed search went through would
            // try what it did (see `restart_of`).
            start = (start + 1).max(matcher.restart);
            if start > subject.len() {
                return Ok(None);
            }
        }
    }

    /// [`Self::find`] with the match handed to `use_match` where the matcher
    /// holds it, so a caller that reads the captures once copies none.
    ///
    /// # Errors
    ///
    /// As [`Self::find`].
    // Kept small so a single run scans in the caller's loop: with the
    // matcher inline, the merged copy of the callers' closures stayed
    // outlined (`string/pattern_gsub` +7% `instructions:u`).
    #[inline]
    pub fn find_with<R>(
        &self,
        subject: &[u8],
        initial: usize,
        use_match: impl FnOnce(&MatchRef<'_>) -> R,
    ) -> Result<Option<R>, PatternError> {
        if let Some(atom) = &self.run_of {
            return Ok(self.find_run(atom, subject, initial).map(|(start, end)| {
                use_match(&MatchRef {
                    start,
                    end,
                    slots: &[],
                })
            }));
        }
        self.find_with_matcher(subject, initial, use_match)
    }

    /// [`Self::find_with`] for a pattern that is not a single run.
    fn find_with_matcher<R>(
        &self,
        subject: &[u8],
        initial: usize,
        use_match: impl FnOnce(&MatchRef<'_>) -> R,
    ) -> Result<Option<R>, PatternError> {
        let mut start = initial.min(subject.len());
        let mut matcher = Matcher {
            subject,
            pattern: self,
            captures: Slots {
                inline: [EMPTY_SLOT; INLINE_CAPTURES],
                spill: if self.captures > INLINE_CAPTURES {
                    vec![EMPTY_SLOT; MAX_CAPTURES]
                } else {
                    Vec::new()
                },
            },
            open: 0,
            steps: 0,
            restart: 0,
        };
        loop {
            if let Some(skip) = &self.skip {
                if self.anchored {
                    if !subject.get(start).is_some_and(|&byte| skip.matches(byte)) {
                        return Ok(None);
                    }
                } else if let Some(offset) = self.first_candidate(skip, &subject[start..]) {
                    start += offset;
                } else {
                    return Ok(None);
                }
            }
            matcher.open = 0;
            matcher.steps = 0;
            matcher.restart = 0;
            if let Some(end) = matcher.run(0, start, 0).map_err(PatternError::from)? {
                let found = MatchRef {
                    start,
                    end,
                    slots: &matcher.captures.all()[..matcher.open],
                };
                return Ok(Some(use_match(&found)));
            }
            if self.anchored || start >= subject.len() {
                return Ok(None);
            }
            // The starts inside the run the failed search went through would
            // try what it did (see `restart_of`).
            start = (start + 1).max(matcher.restart);
            if start > subject.len() {
                return Ok(None);
            }
        }
    }
}

/// The offset of the first `needle` in `bytes`, eight bytes at a time.
// The per-chunk `try_into` compiles against `chunks_exact` slices, not `as_chunks` arrays.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline]
fn position_of_byte(needle: u8, bytes: &[u8]) -> Option<usize> {
    const LOW: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let wanted = LOW * u64::from(needle);
    let mut chunks = bytes.chunks_exact(8);
    let mut base = 0;
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("a chunk of eight")) ^ wanted;
        // The lowest set bit marks the first zero byte of `word`; the bits
        // above it may be wrong, which only the first one is read for.
        let zero = word.wrapping_sub(LOW) & !word & HIGH;
        if zero != 0 {
            return Some(base + (zero.trailing_zeros() / 8) as usize);
        }
        base += 8;
    }
    chunks
        .remainder()
        .iter()
        .position(|&byte| byte == needle)
        .map(|offset| base + offset)
}

/// The offset of the first byte of `bytes` the table marks, eight bytes at a
/// time.
#[inline]
fn position_in_table(table: &[u8; 256], bytes: &[u8]) -> Option<usize> {
    let (chunks, remainder) = bytes.as_chunks::<8>();
    let mut base = 0;
    for chunk in chunks {
        let any = table[usize::from(chunk[0])]
            | table[usize::from(chunk[1])]
            | table[usize::from(chunk[2])]
            | table[usize::from(chunk[3])]
            | table[usize::from(chunk[4])]
            | table[usize::from(chunk[5])]
            | table[usize::from(chunk[6])]
            | table[usize::from(chunk[7])];
        if any != 0 {
            let offset = chunk
                .iter()
                .position(|&byte| table[usize::from(byte)] != 0)
                .expect("a marked byte in the chunk");
            return Some(base + offset);
        }
        base += 8;
    }
    remainder
        .iter()
        .position(|&byte| table[usize::from(byte)] != 0)
        .map(|offset| base + offset)
}

/// Resolves a pattern, minus its anchor, to ops. The pattern passed
/// validation, so syntax errors here would be a bug in `validate`.
fn compile(pattern: &[u8]) -> Result<(Vec<Op>, Vec<u8>), PatternError> {
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    let mut pi = 0;
    while pi < pattern.len() {
        let next = pattern.get(pi + 1).copied();
        match pattern[pi] {
            b'(' if next == Some(b')') => {
                ops.push(Op::Position);
                pi += 2;
            }
            b'(' => {
                ops.push(Op::Open);
                pi += 1;
            }
            b')' => {
                ops.push(Op::Close);
                pi += 1;
            }
            b'$' if pi + 1 == pattern.len() => {
                ops.push(Op::End);
                pi += 1;
            }
            b'%' if next == Some(b'b') => {
                let (Some(&open), Some(&close)) = (pattern.get(pi + 2), pattern.get(pi + 3)) else {
                    return Err(PatternError::MalformedPattern("missing arguments to %b"));
                };
                ops.push(Op::Balance(open, close));
                pi += 4;
            }
            b'%' if next == Some(b'f') => {
                let set = pi + 2;
                if pattern.get(set) != Some(&b'[') {
                    return Err(PatternError::MalformedPattern("missing '[' after %f"));
                }
                let end = class_end(pattern, set)?;
                ops.push(Op::Frontier(set_table(pattern, set)?));
                pi = end;
            }
            b'%' if next.is_some_and(|byte| byte.is_ascii_digit()) => {
                ops.push(Op::Back(pattern[pi + 1] - b'0'));
                pi += 2;
            }
            first => {
                let end = class_end(pattern, pi)?;
                let atom = match first {
                    b'.' => Atom::Any,
                    b'%' => Atom::Set(class_set(pattern[pi + 1])),
                    b'[' => Atom::Set(set_table(pattern, pi)?),
                    literal => Atom::Byte(literal),
                };
                pi = end + 1;
                match pattern.get(end) {
                    Some(b'?') => ops.push(Op::Opt(atom)),
                    Some(b'*') => ops.push(Op::Star(atom)),
                    Some(b'+') => ops.push(Op::Plus(atom)),
                    Some(b'-') => ops.push(Op::Lazy(atom)),
                    _ => {
                        pi = end;
                        if let Atom::Byte(byte) = atom {
                            push_literal(&mut ops, &mut literals, byte);
                        } else {
                            ops.push(Op::One(atom));
                        }
                    }
                }
            }
        }
    }
    Ok((ops, literals))
}

/// Appends a literal byte, extending the run the last op already is.
fn push_literal(ops: &mut Vec<Op>, literals: &mut Vec<u8>, byte: u8) {
    let at = u32::try_from(literals.len()).expect("a pattern is at most 4096 bytes");
    literals.push(byte);
    if let Some(Op::Lit { len, .. }) = ops.last_mut() {
        *len += 1;
    } else {
        ops.push(Op::Lit { start: at, len: 1 });
    }
}

/// The op of a pattern that tells where the search after a failed one may
/// start, with whether the position its run ends at is covered as well.
///
/// A pattern that opens with some `atom*` ops, each apart from a quantified
/// atom `B` that follows (`%s*%a+`, `%s*(%a+)`, `%a-`, `x*`), reaches the
/// rest of the pattern from every start inside a run of `B` through the same
/// positions: the start that fails there fails for the starts after it in the
/// run, which try a subset. The next search can begin after the run. Not for a
/// pattern with a back-reference, whose rest depends on what was captured.
fn restart_of(ops: &[Op]) -> Option<(usize, bool)> {
    if ops.iter().any(|op| matches!(op, Op::Back(_))) {
        return None;
    }
    let stars = ops
        .iter()
        .take_while(|op| matches!(op, Op::Star(_)))
        .count();
    let apart_from = |main: &Atom, before: &[Op]| {
        before.iter().all(|op| match op {
            Op::Star(atom) => match main {
                Atom::Any => false,
                Atom::Byte(byte) => atom.is_apart_from(Skip::Byte(*byte)),
                Atom::Set(set) => atom.is_apart_from(Skip::Set(*set)),
            },
            _ => false,
        })
    };
    // `(atom+)`, `atom+` after the stars, or the last star itself.
    match ops.get(stars) {
        Some(Op::Plus(main) | Op::CapturePlus(main)) if apart_from(main, &ops[..stars]) => {
            return Some((stars, false));
        }
        Some(Op::Lazy(main) | Op::CaptureLazy(main) | Op::CaptureStar(main))
            if apart_from(main, &ops[..stars]) =>
        {
            // The start at the end of the run is covered too only when no
            // star in front could take bytes from there.
            return Some((stars, stars == 0));
        }
        _ => {}
    }
    let last = stars.checked_sub(1)?;
    let Op::Star(main) = &ops[last] else {
        return None;
    };
    apart_from(main, &ops[..last]).then_some((last, last == 0))
}

/// Joins each `(atom+)`, `(atom*)` and `(atom-)` into one op: a capture around
/// a single quantified atom is the commonest capture, and the three ops
/// would recurse into each other for it.
fn fuse_captures(ops: &[Op]) -> Vec<Op> {
    let mut fused = Vec::with_capacity(ops.len());
    let mut index = 0;
    while index < ops.len() {
        if let (Op::Open, Some(&inner), Some(Op::Close)) =
            (ops[index], ops.get(index + 1), ops.get(index + 2))
        {
            let joined = match inner {
                Op::Plus(atom) => Some(Op::CapturePlus(atom)),
                Op::Star(atom) => Some(Op::CaptureStar(atom)),
                Op::Lazy(atom) => Some(Op::CaptureLazy(atom)),
                _ => None,
            };
            if let Some(joined) = joined {
                fused.push(joined);
                index += 3;
                continue;
            }
        }
        fused.push(ops[index]);
        index += 1;
    }
    fused
}

/// The bytes a match can start with, when every match consumes a first byte:
/// the bytes the leading ops that consume (a literal, a class, an atom with
/// `+`) accept, together with those of the optional ops (`*`, `-`, `?`) in
/// front of them, which a match may skip. `None` for a pattern that can match
/// without consuming a byte, or whose first byte depends on something else (a
/// back-reference, `$`, `.`). The scan skips the starts a match cannot begin
/// at.
fn start_filter(ops: &[Op], literals: &[u8]) -> Option<Skip> {
    let mut set = ByteSet::EMPTY;
    let add = |atom: &Atom, set: &mut ByteSet| match atom {
        Atom::Any => false,
        Atom::Byte(byte) => {
            set.insert(*byte);
            true
        }
        Atom::Set(members) => {
            set.union(*members);
            true
        }
    };
    for op in ops {
        match op {
            Op::Open | Op::Position | Op::Close | Op::Frontier(_) => {}
            Op::Star(atom)
            | Op::StarApart(atom)
            | Op::CaptureStar(atom)
            | Op::CaptureLazy(atom)
            | Op::Opt(atom)
            | Op::Lazy(atom)
            | Op::LazyApart(atom) => {
                if !add(atom, &mut set) {
                    return None;
                }
            }
            Op::One(atom) | Op::Plus(atom) | Op::PlusApart(atom) | Op::CapturePlus(atom) => {
                if !add(atom, &mut set) {
                    return None;
                }
                return Some(Skip::of(set));
            }
            Op::Lit { start, .. } => {
                set.insert(literals[*start as usize]);
                return Some(Skip::of(set));
            }
            Op::Balance(open, _) => {
                set.insert(*open);
                return Some(Skip::of(set));
            }
            Op::Back(_) | Op::End => return None,
        }
    }
    None
}

/// What `op` requires of the subject byte at its start, if it must consume
/// one.
fn need_of(op: &Op, literals: &[u8]) -> Option<Skip> {
    match op {
        Op::Lit { start, .. } => Some(Skip::Byte(literals[*start as usize])),
        Op::One(Atom::Set(set)) | Op::Plus(Atom::Set(set)) => Some(Skip::Set(*set)),
        Op::Plus(Atom::Byte(byte)) => Some(Skip::Byte(*byte)),
        _ => None,
    }
}

/// The finished capture a slot holds.
#[inline]
fn capture_of(slot: Slot) -> Capture {
    match slot.end {
        POSITION => Capture {
            start: slot.start,
            end: slot.start,
            position: true,
        },
        OPEN => unreachable!("validation and matching close captures"),
        end => Capture {
            start: slot.start,
            end,
            position: false,
        },
    }
}

/// `end` of a capture that is still open.
const OPEN: usize = usize::MAX;
/// `end` of a position capture, whose `start` is the position.
const POSITION: usize = usize::MAX - 1;

#[derive(Clone, Copy)]
struct Slot {
    start: usize,
    end: usize,
}

const EMPTY_SLOT: Slot = Slot { start: 0, end: 0 };
/// How many captures a match keeps without a heap allocation.
const INLINE_CAPTURES: usize = 8;

/// The captures of a match in progress. Most patterns have a few, so they
/// live in the matcher; initializing room for [`MAX_CAPTURES`] on every
/// search costs more than a short search.
struct Slots {
    inline: [Slot; INLINE_CAPTURES],
    /// Room for [`MAX_CAPTURES`], when the pattern can open more than fit
    /// inline.
    spill: Vec<Slot>,
}

impl Slots {
    #[inline]
    fn all(&self) -> &[Slot] {
        if self.spill.is_empty() {
            &self.inline
        } else {
            &self.spill
        }
    }

    #[inline]
    fn all_mut(&mut self) -> &mut [Slot] {
        if self.spill.is_empty() {
            &mut self.inline
        } else {
            &mut self.spill
        }
    }
}

struct Matcher<'a> {
    subject: &'a [u8],
    pattern: &'a Pattern,
    captures: Slots,
    open: usize,
    steps: usize,
    /// Where a search after a failed one at this start may begin, when the
    /// pattern has a [`Pattern::restart_pc`] and the failed search reached it.
    restart: usize,
}

impl Matcher<'_> {
    /// Whether a continuation that needs `need` of the byte at `at` can match
    /// there.
    #[inline]
    fn viable(&self, need: Option<&Skip>, at: usize) -> bool {
        need.is_none_or(|need| self.subject.get(at).is_some_and(|&byte| need.matches(byte)))
    }

    /// Accounts for a continuation [`Matcher::viable`] ruled out: entering it
    /// would have cost one step and failed at its first byte.
    #[inline]
    fn skipped(&mut self, depth: usize) -> Result<(), Fault> {
        self.steps += 1;
        if self.steps > MAX_STEPS || depth + 1 > MAX_DEPTH {
            return Err(Fault::ResourceLimit);
        }
        Ok(())
    }

    /// [`Self::run`] for the op after a quantifier or a capture. When that is
    /// the end of the pattern the answer is `si` after the step the call would
    /// have counted, so the common pattern that ends in a run does not
    /// recurse for it.
    #[inline(always)]
    fn then(&mut self, pc: usize, si: usize, depth: usize) -> Result<Option<usize>, Fault> {
        if pc < self.pattern.ops.len() {
            return self.run(pc, si, depth);
        }
        self.steps += 1;
        if self.steps > MAX_STEPS || depth > MAX_DEPTH {
            return Err(Fault::ResourceLimit);
        }
        Ok(Some(si))
    }

    /// Matches `ops[pc..]` at `si`. `steps` and `depth` count what the
    /// recursive matcher this replaced counted: one step per item, one depth
    /// per item matched, so the resource limits trip on the same patterns.
    #[allow(clippy::too_many_lines)]
    fn run(
        &mut self,
        mut pc: usize,
        mut si: usize,
        mut depth: usize,
    ) -> Result<Option<usize>, Fault> {
        let subject = self.subject;
        let pattern = self.pattern;
        loop {
            self.steps += 1;
            if self.steps > MAX_STEPS || depth > MAX_DEPTH {
                return Err(Fault::ResourceLimit);
            }
            let Some(op) = pattern.ops.get(pc) else {
                return Ok(Some(si));
            };
            match *op {
                Op::Lit { start, len: 1 } => {
                    // One byte costs the step the loop counted and one depth.
                    if subject.get(si) != Some(&pattern.literals[start as usize]) {
                        return Ok(None);
                    }
                    si += 1;
                    depth += 1;
                    pc += 1;
                }
                Op::Lit { start, len } => {
                    let start = start as usize;
                    let literal = &pattern.literals[start..start + len as usize];
                    let tail = &subject[si..];
                    let matched = if tail.starts_with(literal) {
                        literal.len()
                    } else {
                        literal.iter().zip(tail).take_while(|(a, b)| a == b).count()
                    };
                    // The bytes after the first were one step and one depth
                    // each, up to the one that failed.
                    let extra = if matched < literal.len() {
                        matched
                    } else {
                        literal.len() - 1
                    };
                    self.steps += extra;
                    if self.steps > MAX_STEPS || depth + extra > MAX_DEPTH {
                        return Err(Fault::ResourceLimit);
                    }
                    if matched < literal.len() {
                        return Ok(None);
                    }
                    si += matched;
                    depth += matched;
                    pc += 1;
                }
                Op::One(atom) => {
                    if !subject.get(si).is_some_and(|&byte| atom.matches(byte)) {
                        return Ok(None);
                    }
                    si += 1;
                    depth += 1;
                    pc += 1;
                }
                Op::Opt(atom) => {
                    if subject.get(si).is_some_and(|&byte| atom.matches(byte))
                        && let Some(end) = self.then(pc + 1, si + 1, depth + 1)?
                    {
                        return Ok(Some(end));
                    }
                    depth += 1;
                    pc += 1;
                }
                Op::Star(atom) | Op::Plus(atom) => {
                    if matches!(op, Op::Plus(_))
                        && !subject.get(si).is_some_and(|&byte| atom.matches(byte))
                    {
                        return Ok(None);
                    }
                    let base = si + usize::from(matches!(op, Op::Plus(_)));
                    let mut end = base + atom.run_len(&subject[base..]);
                    if pc == pattern.restart_pc {
                        self.restart = end + usize::from(pattern.restart_past_end);
                    }
                    let need = pattern.needs.get(pc + 1).and_then(Option::as_ref);
                    loop {
                        if self.viable(need, end) {
                            if let Some(found) = self.then(pc + 1, end, depth + 1)? {
                                return Ok(Some(found));
                            }
                        } else {
                            self.skipped(depth)?;
                        }
                        if end == base {
                            return Ok(None);
                        }
                        end -= 1;
                    }
                }
                Op::StarApart(atom) | Op::PlusApart(atom) => {
                    if matches!(op, Op::PlusApart(_))
                        && !subject.get(si).is_some_and(|&byte| atom.matches(byte))
                    {
                        return Ok(None);
                    }
                    let base = si + usize::from(matches!(op, Op::PlusApart(_)));
                    let end = base + atom.run_len(&subject[base..]);
                    if pc == pattern.restart_pc {
                        self.restart = end + usize::from(pattern.restart_past_end);
                    }
                    let need = pattern.needs.get(pc + 1).and_then(Option::as_ref);
                    // Only the end of the run can be followed by the next op:
                    // the shorter ends are skipped, a step each.
                    let shorter = end - base;
                    if self.viable(need, end) {
                        if let Some(found) = self.then(pc + 1, end, depth + 1)? {
                            return Ok(Some(found));
                        }
                    } else {
                        self.skipped(depth)?;
                    }
                    self.steps += shorter;
                    if self.steps > MAX_STEPS || (shorter > 0 && depth + 1 > MAX_DEPTH) {
                        return Err(Fault::ResourceLimit);
                    }
                    return Ok(None);
                }
                Op::LazyApart(atom) => {
                    let need = pattern.needs.get(pc + 1).and_then(Option::as_ref);
                    // The rest of the pattern can only follow where the atom
                    // stops accepting: the positions before are skipped, a
                    // step each.
                    let shorter = atom.run_len(&subject[si..]);
                    si += shorter;
                    if pc == pattern.restart_pc {
                        self.restart = si + usize::from(pattern.restart_past_end);
                    }
                    self.steps += shorter;
                    if self.steps > MAX_STEPS || (shorter > 0 && depth + 1 > MAX_DEPTH) {
                        return Err(Fault::ResourceLimit);
                    }
                    if self.viable(need, si) {
                        if let Some(found) = self.then(pc + 1, si, depth + 1)? {
                            return Ok(Some(found));
                        }
                    } else {
                        self.skipped(depth)?;
                    }
                    return Ok(None);
                }
                Op::Lazy(atom) => {
                    if pc == pattern.restart_pc {
                        self.restart = si
                            + atom.run_len(&subject[si..])
                            + usize::from(pattern.restart_past_end);
                    }
                    let need = pattern.needs.get(pc + 1).and_then(Option::as_ref);
                    loop {
                        if self.viable(need, si) {
                            if let Some(found) = self.then(pc + 1, si, depth + 1)? {
                                return Ok(Some(found));
                            }
                        } else {
                            self.skipped(depth)?;
                        }
                        if !subject.get(si).is_some_and(|&byte| atom.matches(byte)) {
                            return Ok(None);
                        }
                        si += 1;
                    }
                }
                Op::CapturePlus(atom) | Op::CaptureStar(atom) | Op::CaptureLazy(atom) => {
                    // The three ops this stands for, in order: the open (whose
                    // step the loop counted), the quantifier (a step, one
                    // deeper), and for each end it tries the close (a step,
                    // two deeper) and what follows (three deeper).
                    if self.open >= MAX_CAPTURES {
                        return Err(Fault::TooManyCaptures);
                    }
                    self.steps += 1;
                    if self.steps > MAX_STEPS || depth + 1 > MAX_DEPTH {
                        return Err(Fault::ResourceLimit);
                    }
                    let index = self.open;
                    if matches!(op, Op::CaptureLazy(_)) {
                        let mut end = si;
                        if pc == pattern.restart_pc {
                            self.restart = si
                                + atom.run_len(&subject[si..])
                                + usize::from(pattern.restart_past_end);
                        }
                        self.open += 1;
                        loop {
                            self.steps += 1;
                            if self.steps > MAX_STEPS || depth + 2 > MAX_DEPTH {
                                return Err(Fault::ResourceLimit);
                            }
                            self.captures.all_mut()[index] = Slot { start: si, end };
                            if let Some(found) = self.then(pc + 1, end, depth + 3)? {
                                return Ok(Some(found));
                            }
                            if !subject.get(end).is_some_and(|&byte| atom.matches(byte)) {
                                self.open -= 1;
                                return Ok(None);
                            }
                            end += 1;
                        }
                    }
                    let plus = matches!(op, Op::CapturePlus(_));
                    if plus && !subject.get(si).is_some_and(|&byte| atom.matches(byte)) {
                        return Ok(None);
                    }
                    let base = si + usize::from(plus);
                    let mut end = base + atom.run_len(&subject[base..]);
                    if pc == pattern.restart_pc {
                        self.restart = end + usize::from(pattern.restart_past_end);
                    }
                    self.open += 1;
                    loop {
                        self.steps += 1;
                        if self.steps > MAX_STEPS || depth + 2 > MAX_DEPTH {
                            return Err(Fault::ResourceLimit);
                        }
                        self.captures.all_mut()[index] = Slot { start: si, end };
                        if let Some(found) = self.then(pc + 1, end, depth + 3)? {
                            return Ok(Some(found));
                        }
                        if end == base {
                            self.open -= 1;
                            return Ok(None);
                        }
                        end -= 1;
                    }
                }
                Op::Open | Op::Position => {
                    if self.open >= MAX_CAPTURES {
                        return Err(Fault::TooManyCaptures);
                    }
                    let end = if matches!(op, Op::Open) {
                        OPEN
                    } else {
                        POSITION
                    };
                    self.captures.all_mut()[self.open] = Slot { start: si, end };
                    self.open += 1;
                    let found = self.then(pc + 1, si, depth + 1)?;
                    if found.is_none() {
                        self.open -= 1;
                    }
                    return Ok(found);
                }
                Op::Close => {
                    let Some(index) = self.captures.all()[..self.open]
                        .iter()
                        .rposition(|slot| slot.end == OPEN)
                    else {
                        return Err(Fault::InvalidClose);
                    };
                    self.captures.all_mut()[index].end = si;
                    let found = self.then(pc + 1, si, depth + 1)?;
                    if found.is_none() {
                        self.captures.all_mut()[index].end = OPEN;
                    }
                    return Ok(found);
                }
                Op::Back(number) => {
                    let Some(&Slot { start, end }) = self.captures.all()[..self.open]
                        .get(usize::from(number) - 1)
                        .filter(|slot| slot.end != OPEN)
                    else {
                        return Err(Fault::InvalidCapture(number));
                    };
                    // As in Lua, a position capture is a valid index that
                    // never matches.
                    if end == POSITION {
                        return Ok(None);
                    }
                    let captured = &subject[start..end];
                    if !subject[si..].starts_with(captured) {
                        return Ok(None);
                    }
                    si += captured.len();
                    depth += 1;
                    pc += 1;
                }
                Op::Balance(open, close) => {
                    if subject.get(si) != Some(&open) {
                        return Ok(None);
                    }
                    let mut level = 1_usize;
                    let mut cursor = si + 1;
                    loop {
                        let Some(&byte) = subject.get(cursor) else {
                            return Ok(None);
                        };
                        cursor += 1;
                        if byte == close {
                            level -= 1;
                            if level == 0 {
                                break;
                            }
                        } else if byte == open {
                            level += 1;
                        }
                    }
                    si = cursor;
                    depth += 1;
                    pc += 1;
                }
                Op::Frontier(set) => {
                    let previous = if si == 0 { 0 } else { subject[si - 1] };
                    let current = subject.get(si).copied().unwrap_or(0);
                    if set.contains(previous) || !set.contains(current) {
                        return Ok(None);
                    }
                    depth += 1;
                    pc += 1;
                }
                Op::End => return Ok((si == subject.len()).then_some(si)),
            }
        }
    }
}

fn validate(pattern: &[u8]) -> Result<(), PatternError> {
    let mut index = usize::from(pattern.first() == Some(&b'^'));
    let mut opens = 0_usize;
    let mut captures = 0_usize;
    while index < pattern.len() {
        match pattern[index] {
            b'%' => {
                index += 1;
                let Some(byte) = pattern.get(index) else {
                    return Err(PatternError::MalformedPattern("ends with '%'"));
                };
                if *byte == b'b' {
                    if index + 2 >= pattern.len() {
                        return Err(PatternError::MalformedPattern("missing arguments to %b"));
                    }
                    index += 3;
                } else if *byte == b'f' {
                    index += 1;
                    if pattern.get(index) != Some(&b'[') {
                        return Err(PatternError::MalformedPattern("missing '[' after %f"));
                    }
                    index = class_end(pattern, index)?;
                } else if byte.is_ascii_digit()
                    && (*byte == b'0' || usize::from(*byte - b'0') > captures)
                {
                    return Err(PatternError::InvalidCapture(*byte - b'0'));
                } else {
                    index += 1;
                }
            }
            b'[' => index = class_end(pattern, index)?,
            b'(' => {
                captures += 1;
                if captures > MAX_CAPTURES {
                    return Err(PatternError::MalformedPattern("too many captures"));
                }
                opens += 1;
                index += 1;
            }
            b')' => {
                if opens == 0 {
                    return Err(PatternError::MalformedPattern("invalid ')'"));
                }
                opens -= 1;
                index += 1;
            }
            _ => index += 1,
        }
    }
    if opens != 0 {
        Err(PatternError::MalformedPattern("unfinished capture"))
    } else {
        Ok(())
    }
}

fn class_end(pattern: &[u8], pi: usize) -> Result<usize, PatternError> {
    match pattern.get(pi) {
        Some(b'%') => {
            if pi + 1 >= pattern.len() {
                Err(PatternError::MalformedPattern("ends with '%'"))
            } else {
                Ok(pi + 2)
            }
        }
        Some(b'[') => {
            let mut index = pi + 1;
            if pattern.get(index) == Some(&b'^') {
                index += 1;
            }
            if pattern.get(index) == Some(&b']') {
                index += 1;
            }
            while let Some(byte) = pattern.get(index) {
                if *byte == b']' {
                    return Ok(index + 1);
                }
                if *byte == b'%' && index + 1 < pattern.len() {
                    index += 2;
                } else {
                    index += 1;
                }
            }
            Err(PatternError::MalformedPattern("missing ']'"))
        }
        Some(_) => Ok(pi + 1),
        None => Err(PatternError::MalformedPattern("unexpected end")),
    }
}

#[cfg(test)]
fn match_set(byte: u8, pattern: &[u8], start: usize) -> Result<bool, PatternError> {
    let mut index = start + 1;
    let complement = pattern.get(index) == Some(&b'^');
    if complement {
        index += 1;
    }
    let mut matched = false;
    // Lua accepts `]` as the first member of a set, including after `^`,
    // and as the start of a range there (`[]-a]`).
    let first = index;
    while let Some(&current) = pattern.get(index) {
        if current == b']' && index != first {
            return Ok(if complement { !matched } else { matched });
        }
        if current == b'%' {
            let class = *pattern
                .get(index + 1)
                .ok_or(PatternError::MalformedPattern("missing class"))?;
            matched |= match_class(byte, class);
            index += 2;
        } else if pattern.get(index + 1) == Some(&b'-')
            && pattern.get(index + 2).is_some_and(|next| *next != b']')
        {
            let end = pattern[index + 2];
            matched |= current <= byte && byte <= end;
            index += 3;
        } else {
            matched |= byte == current;
            index += 1;
        }
    }
    Err(PatternError::MalformedPattern("missing ']'"))
}

#[cfg(test)]
fn match_class(byte: u8, class: u8) -> bool {
    let lower = class.to_ascii_lowercase();
    let matched = match lower {
        b'a' => byte.is_ascii_alphabetic(),
        b'c' => byte.is_ascii_control(),
        b'd' => byte.is_ascii_digit(),
        b'g' => byte.is_ascii_graphic(),
        b'l' => byte.is_ascii_lowercase(),
        b'p' => byte.is_ascii_punctuation(),
        b's' => byte.is_ascii_whitespace() || byte == 11,
        b'u' => byte.is_ascii_uppercase(),
        b'w' => byte.is_ascii_alphanumeric(),
        b'x' => byte.is_ascii_hexdigit(),
        b'z' => byte == 0,
        // Any other byte stands for itself, whatever its case.
        _ => return byte == class,
    };
    if class.is_ascii_uppercase() {
        !matched
    } else {
        matched
    }
}

#[cfg(test)]
mod reference {
    use super::*;

    #[derive(Clone, Debug, Default)]
    struct State {
        captures: Vec<CaptureState>,
    }
    #[derive(Clone, Debug)]
    enum CaptureState {
        Open(usize),
        Complete {
            start: usize,
            end: usize,
            position: bool,
        },
    }

    #[derive(Default)]
    struct Budget {
        steps: usize,
    }
    impl Budget {
        fn tick(&mut self, depth: usize) -> Result<(), PatternError> {
            self.steps += 1;
            if self.steps > MAX_STEPS || depth > MAX_DEPTH {
                Err(PatternError::ResourceLimit)
            } else {
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn match_here(
        subject: &[u8],
        pattern: &[u8],
        si: usize,
        pi: usize,
        mut state: State,
        budget: &mut Budget,
        depth: usize,
    ) -> Result<Option<(usize, State)>, PatternError> {
        budget.tick(depth)?;
        if pi == pattern.len() {
            return Ok(Some((si, state)));
        }
        match pattern[pi] {
            b'(' => {
                if pattern.get(pi + 1) == Some(&b')') {
                    state.captures.push(CaptureState::Complete {
                        start: si,
                        end: si,
                        position: true,
                    });
                    return match_here(subject, pattern, si, pi + 2, state, budget, depth + 1);
                }
                if state.captures.len() >= MAX_CAPTURES {
                    return Err(PatternError::MalformedPattern("too many captures"));
                }
                state.captures.push(CaptureState::Open(si));
                return match_here(subject, pattern, si, pi + 1, state, budget, depth + 1);
            }
            b')' => {
                let Some(index) = state
                    .captures
                    .iter()
                    .rposition(|capture| matches!(capture, CaptureState::Open(_)))
                else {
                    return Err(PatternError::MalformedPattern("invalid ')'"));
                };
                let CaptureState::Open(start) = state.captures[index] else {
                    unreachable!()
                };
                state.captures[index] = CaptureState::Complete {
                    start,
                    end: si,
                    position: false,
                };
                return match_here(subject, pattern, si, pi + 1, state, budget, depth + 1);
            }
            b'$' if pi + 1 == pattern.len() => {
                return Ok((si == subject.len()).then_some((si, state)));
            }
            b'%' if pattern.get(pi + 1) == Some(&b'b') => {
                let Some(&open) = pattern.get(pi + 2) else {
                    return Err(PatternError::MalformedPattern("missing arguments to %b"));
                };
                let Some(&close) = pattern.get(pi + 3) else {
                    return Err(PatternError::MalformedPattern("missing arguments to %b"));
                };
                if subject.get(si) != Some(&open) {
                    return Ok(None);
                }
                let mut level = 1_usize;
                let mut cursor = si + 1;
                while cursor < subject.len() {
                    if subject[cursor] == close {
                        level -= 1;
                        if level == 0 {
                            return match_here(
                                subject,
                                pattern,
                                cursor + 1,
                                pi + 4,
                                state,
                                budget,
                                depth + 1,
                            );
                        }
                    } else if subject[cursor] == open {
                        level += 1;
                    }
                    cursor += 1;
                }
                return Ok(None);
            }
            b'%' if pattern.get(pi + 1) == Some(&b'f') => {
                let set = pi + 2;
                if pattern.get(set) != Some(&b'[') {
                    return Err(PatternError::MalformedPattern("missing '[' after %f"));
                }
                let end = class_end(pattern, set)?;
                let previous = if si == 0 { 0 } else { subject[si - 1] };
                let current = subject.get(si).copied().unwrap_or(0);
                if !match_set(previous, pattern, set)? && match_set(current, pattern, set)? {
                    return match_here(subject, pattern, si, end, state, budget, depth + 1);
                }
                return Ok(None);
            }
            b'%' if pattern.get(pi + 1).is_some_and(u8::is_ascii_digit) => {
                let index = usize::from(pattern[pi + 1] - b'1');
                let Some(CaptureState::Complete {
                    start,
                    end,
                    position,
                }) = state.captures.get(index)
                else {
                    return Err(PatternError::InvalidCapture(pattern[pi + 1] - b'0'));
                };
                if *position {
                    return Ok(None);
                }
                let captured = &subject[*start..*end];
                if subject
                    .get(si..)
                    .is_some_and(|tail| tail.starts_with(captured))
                {
                    return match_here(
                        subject,
                        pattern,
                        si + captured.len(),
                        pi + 2,
                        state,
                        budget,
                        depth + 1,
                    );
                }
                return Ok(None);
            }
            _ => {}
        }
        let ep = class_end(pattern, pi)?;
        let suffix = pattern.get(ep).copied();
        match suffix {
            Some(b'?') => {
                if single_match(subject, si, pattern, pi)?
                    && let Some(found) = match_here(
                        subject,
                        pattern,
                        si + 1,
                        ep + 1,
                        state.clone(),
                        budget,
                        depth + 1,
                    )?
                {
                    return Ok(Some(found));
                }
                match_here(subject, pattern, si, ep + 1, state, budget, depth + 1)
            }
            Some(b'*') => max_expand(subject, pattern, si, pi, ep + 1, state, budget, depth),
            Some(b'+') => {
                if single_match(subject, si, pattern, pi)? {
                    max_expand(subject, pattern, si + 1, pi, ep + 1, state, budget, depth)
                } else {
                    Ok(None)
                }
            }
            Some(b'-') => min_expand(subject, pattern, si, pi, ep + 1, state, budget, depth),
            _ => {
                if single_match(subject, si, pattern, pi)? {
                    match_here(subject, pattern, si + 1, ep, state, budget, depth + 1)
                } else {
                    Ok(None)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    fn max_expand(
        subject: &[u8],
        pattern: &[u8],
        si: usize,
        atom: usize,
        rest: usize,
        state: State,
        budget: &mut Budget,
        depth: usize,
    ) -> Result<Option<(usize, State)>, PatternError> {
        let mut end = si;
        while single_match(subject, end, pattern, atom)? {
            end += 1;
        }
        loop {
            if let Some(found) = match_here(
                subject,
                pattern,
                end,
                rest,
                state.clone(),
                budget,
                depth + 1,
            )? {
                return Ok(Some(found));
            }
            if end == si {
                return Ok(None);
            }
            end -= 1;
        }
    }
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    fn min_expand(
        subject: &[u8],
        pattern: &[u8],
        mut si: usize,
        atom: usize,
        rest: usize,
        state: State,
        budget: &mut Budget,
        depth: usize,
    ) -> Result<Option<(usize, State)>, PatternError> {
        loop {
            if let Some(found) =
                match_here(subject, pattern, si, rest, state.clone(), budget, depth + 1)?
            {
                return Ok(Some(found));
            }
            if !single_match(subject, si, pattern, atom)? {
                return Ok(None);
            }
            si += 1;
        }
    }

    fn single_match(
        subject: &[u8],
        si: usize,
        pattern: &[u8],
        pi: usize,
    ) -> Result<bool, PatternError> {
        let Some(&byte) = subject.get(si) else {
            return Ok(false);
        };
        match pattern[pi] {
            b'.' => Ok(true),
            b'%' => Ok(match_class(
                byte,
                *pattern
                    .get(pi + 1)
                    .ok_or(PatternError::MalformedPattern("ends with '%'"))?,
            )),
            b'[' => match_set(byte, pattern, pi),
            expected => Ok(byte == expected),
        }
    }

    /// The matcher before pattern compilation: raw pattern bytes, a cloned
    /// capture list per backtrack.
    pub(super) fn find(
        bytes: &[u8],
        anchors: bool,
        subject: &[u8],
        initial: usize,
    ) -> Result<Option<Match>, PatternError> {
        validate(bytes)?;
        let anchored = anchors && bytes.first() == Some(&b'^');
        let pattern = &bytes[usize::from(anchored)..];
        let mut start = initial.min(subject.len());
        loop {
            let mut budget = Budget::default();
            if let Some((end, state)) =
                match_here(subject, pattern, start, 0, State::default(), &mut budget, 0)?
            {
                let captures = state
                    .captures
                    .iter()
                    .map(|capture| match capture {
                        CaptureState::Complete {
                            start,
                            end,
                            position,
                        } => Capture {
                            start: *start,
                            end: *end,
                            position: *position,
                        },
                        CaptureState::Open(_) => unreachable!(),
                    })
                    .collect();
                return Ok(Some(Match {
                    start,
                    end,
                    captures,
                }));
            }
            if anchored || start == subject.len() {
                return Ok(None);
            }
            start += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn whitespace_class_matches_lua_c_locale_including_vertical_tab() {
        let whitespace = Pattern::new(b"^%s$").unwrap();
        let complement = Pattern::new(b"^%S$").unwrap();
        let set = Pattern::new(b"^[%s]$").unwrap();
        for byte in 0..=255 {
            let expected = matches!(byte, b'\t' | b'\n' | 11 | b'\x0c' | b'\r' | b' ');
            assert_eq!(
                whitespace.find(&[byte], 0).unwrap().is_some(),
                expected,
                "byte {byte}"
            );
            assert_eq!(
                set.find(&[byte], 0).unwrap().is_some(),
                expected,
                "set byte {byte}"
            );
            assert_eq!(
                complement.find(&[byte], 0).unwrap().is_some(),
                !expected,
                "complement byte {byte}"
            );
        }
    }
    use super::*;

    #[test]
    fn gmatch_patterns_match_a_leading_caret_literally() {
        let pattern = Pattern::new_for_gmatch(b"^a").unwrap();
        assert!(!pattern.is_anchored());
        assert!(pattern.find(b"aaa", 0).unwrap().is_none());
        assert_eq!(pattern.find(b"x^a", 0).unwrap().unwrap().start(), 1);
    }

    #[test]
    fn is_anchored_matches_leading_caret() {
        assert!(Pattern::new(b"^abc").unwrap().is_anchored());
        assert!(!Pattern::new(b"abc^").unwrap().is_anchored());
        assert!(!Pattern::new(b"%^abc").unwrap().is_anchored());
        assert!(!Pattern::new(b"").unwrap().is_anchored());
    }

    fn one(subject: &[u8], pattern: &[u8]) -> Match {
        Pattern::new(pattern)
            .unwrap()
            .find(subject, 0)
            .unwrap()
            .unwrap()
    }
    #[test]
    fn classes_sets_and_anchors() {
        assert_eq!(one(b"ab 42", b"%d+").start(), 3);
        assert!(
            Pattern::new(b"^[a-z]+$")
                .unwrap()
                .find(b"word", 0)
                .unwrap()
                .is_some()
        );
        assert!(
            Pattern::new(b"[^a]")
                .unwrap()
                .find(b"b", 0)
                .unwrap()
                .is_some()
        );
    }
    #[test]
    fn captures_backrefs_and_positions() {
        let found = one(b"abab", b"(%a+)%1");
        assert_eq!((found.start(), found.end()), (0, 4));
        let found = one(b"ab", b"a()b");
        assert_eq!(
            found.captures()[0],
            Capture {
                start: 1,
                end: 1,
                position: true
            }
        );
    }
    #[test]
    fn balanced_frontier_and_minimal() {
        assert!(
            Pattern::new(b"%b()$")
                .unwrap()
                .find(b"(a(b))", 0)
                .unwrap()
                .is_some()
        );
        assert_eq!(one(b"a123b456b", b"a.-b").end(), 5);
        assert_eq!(one(b"!word?", b"%f[%a]word").start(), 1);
    }
    #[test]
    fn arbitrary_bytes_and_empty_progress() {
        assert!(
            Pattern::new(b"%z")
                .unwrap()
                .find(b"a\0b", 0)
                .unwrap()
                .is_some()
        );
        assert_eq!(one(b"a", b"a?").end(), 1);
    }
    #[test]
    fn first_byte_filter_preserves_later_and_empty_matches() {
        assert_eq!(one(b"abc 42", b"%d+").start(), 4);
        assert_eq!(one(b"\0a\0ab", b"ab").start(), 3);
        assert!(
            Pattern::new(b"^%d+")
                .unwrap()
                .find(b"a1", 0)
                .unwrap()
                .is_none()
        );
        assert_eq!(one(b"abc", b"%d*").start(), 0);
        assert_eq!(one(b"abc", b"x?").start(), 0);
    }
    #[test]
    fn official_pm_core_cases() {
        let find = |subject: &[u8], pattern: &[u8]| {
            Pattern::new(pattern).unwrap().find(subject, 0).unwrap()
        };
        assert_eq!(find(b"a\0a\0a\0a\0\0ab", b"\0ab").unwrap().start(), 8);
        assert!(find(b"alo123alo", b"^12").is_none());
        assert!(find(b"function", b"%f[\x01-\xff]%w").is_some());
        assert_eq!(find(b"]]]x", b"[^]]").unwrap().start(), 3);
        // A leading `]` can start a range, as in lstrlib's matchbracketclass.
        assert!(find(b"^", b"[]-a]").is_some());
        assert!(find(b"-", b"[]-a]").is_none());
        assert!(find(b"-", b"[^]-a]").is_some());
        // A back-reference to a position capture is valid and never matches.
        assert!(find(b"x", b"()%1").is_none());
        assert!(find(b"(a(b))", b"%b()$").is_some());
        assert_eq!(find(b"alo 'oi' alo", b"%b''").unwrap().start(), 4);
        let matched = find(b"abc=xyz", b"(%w*)(%p)(%w+)").unwrap();
        assert_eq!(matched.captures().len(), 3);
        assert_eq!(
            &b"abc=xyz"[matched.captures()[2].start()..matched.captures()[2].end()],
            b"xyz"
        );
    }
    #[test]
    fn errors_read_as_stock_lua_spells_them() {
        let message =
            |pattern: &[u8], legacy: bool| Pattern::new(pattern).unwrap_err().lua_message(legacy);
        for legacy in [false, true] {
            assert_eq!(message(b"[a", legacy), "malformed pattern (missing ']')");
            assert_eq!(message(b"%", legacy), "malformed pattern (ends with '%')");
            assert_eq!(message(b"(a", legacy), "unfinished capture");
            assert_eq!(message(b"a)", legacy), "invalid pattern capture");
            assert_eq!(message(b"%fa", legacy), "missing '[' after '%f' in pattern");
        }
        assert_eq!(
            message(b"%b", false),
            "malformed pattern (missing arguments to '%b')"
        );
        assert_eq!(message(b"%b", true), "unbalanced pattern");
        assert_eq!(message(b"(a)%2", false), "invalid capture index %2");
        assert_eq!(message(b"(a)%2", true), "invalid capture index");
        assert_eq!(message(b"%1", false), "invalid capture index %1");
    }
    #[test]
    fn rejects_bad_patterns() {
        assert!(Pattern::new(b"(").is_err());
        assert!(Pattern::new(b"%b(").is_err());
        assert!(Pattern::new(b"%1").is_err());
    }
    #[test]
    fn a_shared_pattern_is_built_once_and_matches_like_a_new_one() {
        let first = Pattern::shared(b"(%d+)-(%a)", false).unwrap();
        let second = Pattern::shared(b"(%d+)-(%a)", false).unwrap();
        assert!(Rc::ptr_eq(&first, &second));
        let fresh = Pattern::new(b"(%d+)-(%a)").unwrap();
        assert_eq!(
            first.find(b"x 12-a", 0).unwrap(),
            fresh.find(b"x 12-a", 0).unwrap()
        );
        assert_eq!(first.source(), b"(%d+)-(%a)");
        // The kinds are apart: a `gmatch` pattern's caret is a character.
        let anchored = Pattern::shared(b"^a", false).unwrap();
        let literal = Pattern::shared(b"^a", true).unwrap();
        assert!(anchored.is_anchored());
        assert!(!literal.is_anchored());
        assert!(anchored.find(b"ba", 0).unwrap().is_none());
        assert!(literal.find(b"b^a", 0).unwrap().is_some());
    }
    #[test]
    fn a_shared_pattern_that_is_bad_or_long_is_not_kept_wrongly() {
        assert!(Pattern::shared(b"(", false).is_err());
        assert!(Pattern::shared(b"(", false).is_err());
        let long = vec![b'a'; CACHED_PATTERN_BYTES + 1];
        let one = Pattern::shared(&long, false).unwrap();
        let two = Pattern::shared(&long, false).unwrap();
        assert!(!Rc::ptr_eq(&one, &two));
        // A full cache starts over and still answers.
        for number in 0..(CACHED_PATTERNS + 5) {
            let text = format!("x{number}");
            assert!(Pattern::shared(text.as_bytes(), false).is_ok());
        }
        assert!(Pattern::shared(b"x1", false).is_ok());
    }

    fn agrees(pattern: &[u8], subject: &[u8]) {
        for anchors in [true, false] {
            let compiled = Pattern::build(pattern, anchors);
            let Ok(compiled) = compiled else {
                assert_eq!(
                    compiled.err(),
                    validate(pattern).err(),
                    "build error of {:?}",
                    String::from_utf8_lossy(pattern)
                );
                continue;
            };
            for initial in [0, 1, subject.len() / 2, subject.len(), subject.len() + 3] {
                assert_eq!(
                    compiled.find(subject, initial),
                    reference::find(pattern, anchors, subject, initial),
                    "pattern {:?} subject {:?} init {initial} anchors {anchors}",
                    String::from_utf8_lossy(pattern),
                    String::from_utf8_lossy(subject),
                );
            }
        }
    }

    const SUBJECTS: &[&[u8]] = &[
        b"",
        b"a",
        b"key_1=Value42 next",
        b"  xHello World 2024!",
        b"MIXED case_text, 7 items",
        b"a=1 b=22 c=333",
        b"THE (quick) [brown] {fox} f(a(b)c) end",
        b"aaa aab abab abba 0x1F \x00\0 \t\x0b",
        b"2024-01-17 INFO [net] connected to 10.0.0.1 after 3 tries",
        b"=====....-----    ",
        b"\x00\x01 \xff\xfe z Z",
    ];

    #[test]
    fn compiled_program_agrees_with_the_raw_pattern_matcher_on_the_benchmark_patterns() {
        let prefixes = ["", "^", "x", "%s*"];
        let classes = ["%a", "%d", "%l", "%u", "%w", "%p", "[aeiou]", "[%d_]"];
        let quantifiers = ["", "+", "*", "-"];
        let suffixes = ["", "%d", "(%a+)", "()", "=", "%f[%W]", "[^%s]?", "%s"];
        for prefix in prefixes {
            for class in classes {
                for quantifier in quantifiers {
                    for suffix in suffixes {
                        let pattern = format!("{prefix}{class}{quantifier}{suffix}");
                        for subject in SUBJECTS {
                            agrees(pattern.as_bytes(), subject);
                        }
                    }
                }
            }
        }
        for pattern in ["^(%d+)%-(%d+)%-(%d+) (%u+) ", "%[(%w+)%]", "%a+", "%d+"] {
            for subject in SUBJECTS {
                agrees(pattern.as_bytes(), subject);
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn compiled_program_agrees_with_the_raw_pattern_matcher_on_edge_cases() {
        let patterns: &[&[u8]] = &[
            b"",
            b"^",
            b"$",
            b"^$",
            b"a$",
            b"$a",
            b"a^",
            b"^^a",
            b"%^a",
            b"a%$",
            b".",
            b".-",
            b".*",
            b".+",
            b".?",
            b".-$",
            b"^.-$",
            b"%f[%w]%w+",
            b"%f[%W]",
            b"%f[^%w]",
            b"%f[a-c]",
            b"%f[\0]",
            b"%f[%z]",
            b"%b()",
            b"%b[]",
            b"%bxy",
            b"%b()%b[]",
            b"(%b())",
            b"%b()*",
            b"(a)%1",
            b"(a*)%1",
            b"(.)%1",
            b"(.-)%1",
            b"((a)(b))%2%3",
            b"(a)(b)%2%1",
            b"()",
            b"()a()",
            b"(()a)",
            b"()%1",
            b"(a)()%2",
            b"a-",
            b"a-b",
            b"a*b",
            b"a+b",
            b"a?b",
            b"a?",
            b"a-$",
            b"(a-)b",
            b"--",
            b"-",
            b"*",
            b"+",
            b"(*)",
            b"(a*(.)%w(%s*))",
            b"[a-c]",
            b"[^a-c]",
            b"[a-]",
            b"[-a]",
            b"[]]",
            b"[^]]",
            b"[]-a]",
            b"[^]-a]",
            b"[]a]+",
            b"[a%-z]",
            b"[%a_]+",
            b"[%d-z]",
            b"[a-%d]",
            b"[%]]",
            b"[^%s]+",
            b"[a-cx-z]+",
            b"[z-a]",
            b"[%w_%.]+",
            b"%z",
            b"%Z+",
            b"[%z]",
            b"%Q",
            b"%q+",
            b"%.",
            b"%%",
            b"%-",
            b"%a%A",
            b"%c%C",
            b"%g%G",
            b"%p%P",
            b"%s%S",
            b"%x%X",
            b"%l%L",
            b"%u%U",
            b"%w%W",
            b"%d%D",
            b"aaaa",
            b"abab",
            b"(ab)+",
            b"ab(",
            b"ab)",
            b"%",
            b"[a",
            b"%b",
            b"%bx",
            b"%f",
            b"%fa",
            b"%0",
            b"%1",
            b"(a)%2",
            b"(a%1)",
            b"[a-",
            b"[]",
            b"((((((((((((((((((((((((((((((((a))))))))))))))))))))))))))))))))",
            b"(((((((((((((((((((((((((((((((((a)))))))))))))))))))))))))))))))))",
            b"(.-)%s*=%s*(.-)%s*$",
            b"^%s*(.-)%s*$",
            b"^(%w+)=(%w*)$",
            b"(%w+)=(%w+)",
            b".-.-.-.-x",
            b"(.*)(.*)(.*)(.*)x",
            b"a*a*a*a*b",
        ];
        for pattern in patterns {
            for subject in SUBJECTS {
                agrees(pattern, subject);
            }
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn compiled_program_agrees_with_the_raw_pattern_matcher_on_random_patterns() {
        let pieces: &[&[u8]] = &[
            b"a", b"b", b"x", b"1", b" ", b".", b"%a", b"%d", b"%s", b"%w", b"%W", b"%p", b"%z",
            b"[ab]", b"[^ab]", b"[a-c]", b"[%d_]", b"[^%s]", b"(", b")", b"()", b"%1", b"%2",
            b"%b()", b"%f[%w]", b"%f[%W]", b"*", b"+", b"-", b"?", b"$", b"^", b"=", b"%.", b"%%",
        ];
        let alphabet = b"ab x1=()_.%\0\xffAB";
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..4000 {
            let mut pattern = Vec::new();
            for _ in 0..(next() % 7) {
                pattern.extend_from_slice(pieces[(next() % pieces.len() as u64) as usize]);
            }
            let mut subject = Vec::new();
            for _ in 0..(next() % 14) {
                subject.push(alphabet[(next() % alphabet.len() as u64) as usize]);
            }
            agrees(&pattern, &subject);
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn leading_runs_and_captured_runs_agree_with_the_raw_pattern_matcher() {
        // Patterns of the shapes the matcher joins or restarts after:
        // `atom*` ops, a quantified (maybe captured) atom, and a tail.
        let atoms: &[&[u8]] = &[
            b"%s", b"%a", b"%d", b"%w", b"x", b".", b"[ab]", b"[^ab]", b"%p",
        ];
        let quantifiers: &[&[u8]] = &[b"*", b"+", b"-", b""];
        let tails: &[&[u8]] = &[
            b"", b"x", b"%d", b"=", b" ", b"%f[%W]", b"(%a+)", b"()", b"$", b"%s*$", b"[^%s]?",
            b"%1", b"(.-)=",
        ];
        let alphabet = b"ab x1=_.  \xff";
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..6000 {
            let mut pattern = Vec::new();
            for _ in 0..(next() % 3) {
                pattern.extend_from_slice(atoms[(next() % atoms.len() as u64) as usize]);
                pattern.push(b'*');
            }
            let captured = next() % 2 == 0;
            if captured {
                pattern.push(b'(');
            }
            pattern.extend_from_slice(atoms[(next() % atoms.len() as u64) as usize]);
            pattern.extend_from_slice(quantifiers[(next() % quantifiers.len() as u64) as usize]);
            if captured {
                pattern.push(b')');
            }
            pattern.extend_from_slice(tails[(next() % tails.len() as u64) as usize]);
            let mut subject = Vec::new();
            for _ in 0..(next() % 18) {
                subject.push(alphabet[(next() % alphabet.len() as u64) as usize]);
            }
            agrees(&pattern, &subject);
        }
    }

    #[test]
    fn captured_runs_trip_the_same_limits_as_the_raw_pattern_matcher() {
        let subject = [b'a'; 3000];
        for pattern in [
            &b"(a+)b"[..],
            b"(a*)b",
            b"(a-)b",
            b"(a*)(a*)(a*)b",
            b"(a+)(a+)(a+)(a+)b",
            b"%s*(a+)b",
            b"a*(a*)a*b",
        ] {
            for length in [10, 60] {
                agrees(pattern, &subject[..length]);
            }
        }
    }

    #[test]
    fn compiled_program_trips_the_same_limits_as_the_raw_pattern_matcher() {
        // One depth per item: 600 literal items overrun MAX_DEPTH only when
        // they are matched, not when an early byte already failed.
        let long = vec![b'a'; 600];
        agrees(&long, &long);
        agrees(&long, b"aaa");
        agrees(&long, b"baaaa");
        let mut mismatch_late = vec![b'a'; 520];
        mismatch_late.push(b'b');
        agrees(&mismatch_late, &[b'a'; 600]);
        agrees(&mismatch_late, &[b'a'; 510]);
        let mut sets = Vec::new();
        for _ in 0..530 {
            sets.extend_from_slice(b"%a");
        }
        agrees(&sets, &[b'q'; 600]);
        agrees(&sets, &[b'q'; 100]);
        let subject = [b'a'; 3000];
        agrees(b"(a*)*b", &subject);
        agrees(b"a*a*a*a*a*a*b", &subject[..200]);
        agrees(b".-.-.-.-.-.-b", &subject[..200]);
        assert_eq!(
            Pattern::new(b"a*a*a*a*a*a*b")
                .unwrap()
                .find(&subject[..400], 0),
            Err(PatternError::ResourceLimit)
        );
        assert_eq!(
            Pattern::new(&long).unwrap().find(&long, 0),
            Err(PatternError::ResourceLimit)
        );
    }

    #[test]
    fn leading_optional_ops_filter_the_starts_without_hiding_a_match() {
        let patterns: &[&[u8]] = &[
            b"%s*%a+%d",
            b"%s*%a+=",
            b"x*y",
            b"x-y",
            b"x?y",
            b"a?b?c",
            b"(x*)y",
            b"()x*y",
            b"%f[%a]x*y",
            b"%s*(%a+)",
            b"^%s*%a+",
            b"^x*y",
            b"x*",
            b"x*$",
            b".*x",
            b".-x",
            b"(a*)b%1",
            b"%s*%b()",
            b"-?%d+",
            b"[+-]?%d+%.?%d*",
        ];
        for pattern in patterns {
            for subject in SUBJECTS {
                agrees(pattern, subject);
            }
        }
        assert_eq!(one(b"  Hello 42", b"%s*%a+%s*%d").start(), 0);
        assert_eq!(one(b"..;;x", b"x*y?x").start(), 4);
        assert_eq!(one(b"abc =12", b"%s*=%d+").start(), 3);
        assert_eq!(one(b"zzzy", b"x*y").start(), 3);
        assert!(
            Pattern::new(b"%s*%a+%d")
                .unwrap()
                .find(b"  ...!!", 0)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_run_the_next_op_cannot_start_in_trips_the_same_limits() {
        // The shorter ends of such a run are skipped in one go, and counted
        // as the steps they would have taken: a run of more than MAX_STEPS
        // trips the limit as the one-by-one backtracking did.
        let long = vec![b'a'; MAX_STEPS + 100];
        for pattern in [&b"a*b"[..], b"a+b", b"a-b", b"a*=", b"a-%d", b"%a+%d"] {
            assert_eq!(
                Pattern::new(pattern).unwrap().find(&long, 0),
                Err(PatternError::ResourceLimit),
                "{}",
                String::from_utf8_lossy(pattern)
            );
        }
        // Below the limit the same patterns fail or match as before, on
        // subjects the raw matcher can still follow.
        for length in [3, 40, 300, 511, 512, 513, 600] {
            let run = vec![b'a'; length];
            let mut tail = run.clone();
            tail.push(b'b');
            for pattern in [
                &b"a*b"[..],
                b"a+b",
                b"a-b",
                b"a*=",
                b"a-%d",
                b"%a+%d",
                b"%a+b",
            ] {
                agrees(pattern, &run);
                agrees(pattern, &tail);
            }
        }
    }

    #[test]
    fn find_with_hands_over_the_match_find_would_give() {
        let prefixes = ["", "^", "x", "%s*"];
        let classes = ["%a", "%d", "[aeiou]"];
        let quantifiers = ["", "+", "-"];
        let suffixes = ["", "(%a+)", "()", "=", "(%d)(%a)()"];
        for prefix in prefixes {
            for class in classes {
                for quantifier in quantifiers {
                    for suffix in suffixes {
                        let pattern =
                            Pattern::new(format!("{prefix}{class}{quantifier}{suffix}")).unwrap();
                        for subject in SUBJECTS {
                            for initial in [0, 3] {
                                let found = pattern.find(subject, initial).unwrap();
                                let with = pattern
                                    .find_with(subject, initial, |found| {
                                        (
                                            found.start(),
                                            found.end(),
                                            (0..found.capture_count())
                                                .map(|index| found.capture(index))
                                                .collect::<Vec<_>>(),
                                        )
                                    })
                                    .unwrap();
                                assert_eq!(
                                    found.map(|found| (
                                        found.start(),
                                        found.end(),
                                        found.captures().to_vec()
                                    )),
                                    with
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unknown_escapes_match_themselves_whatever_their_case() {
        assert_eq!(one(b"xQx", b"%Q").start(), 1);
        assert!(
            Pattern::new(b"%Q")
                .unwrap()
                .find(b"abc", 0)
                .unwrap()
                .is_none()
        );
        assert_eq!(one(b"a\0b", b"%z").start(), 1);
        assert_eq!(one(b"a\0b", b"%Z+").start(), 0);
    }

    #[test]
    fn shared_finds_a_pattern_again_after_its_address_is_reused() {
        let first = Pattern::shared(b"%d+x", false).unwrap();
        let mut owned = b"%d+x".to_vec();
        let again = Pattern::shared(&owned, false).unwrap();
        assert!(Rc::ptr_eq(&first, &again));
        owned[3] = b'y';
        assert_eq!(Pattern::shared(&owned, false).unwrap().source(), b"%d+y");
    }

    #[test]
    fn a_single_run_has_a_class_table_and_nothing_else_does() {
        let run: &[(&[u8], bool, bool)] = &[
            (b"%d+", false, false),
            (b"^%d+", true, false),
            (b"[%w_]+", false, false),
            (b"[^, ]+", false, false),
            (b".+", false, false),
            (b"a+", false, false),
            (b"(%d+)", false, true),
            (b"^([%a_]+)", true, true),
        ];
        for (source, anchored, captured) in run {
            let pattern = Pattern::new(source).unwrap();
            let (table, shape) = pattern.run_class().expect("a single run");
            assert_eq!(shape.anchored, *anchored, "{source:?}");
            assert_eq!(shape.captured, *captured, "{source:?}");
            // The table scans as the matcher does, for every start.
            for subject in [&b"ab 12,x_9\0\x80 z"[..], b"", b"\xff", b"123"] {
                for initial in 0..=subject.len() + 1 {
                    let mut expected = None;
                    let from = initial.min(subject.len());
                    let starts = if *anchored {
                        from..from + 1
                    } else {
                        from..subject.len()
                    };
                    for start in starts {
                        if subject
                            .get(start)
                            .is_some_and(|&b| table[usize::from(b)] != 0)
                        {
                            let mut end = start + 1;
                            while subject
                                .get(end)
                                .is_some_and(|&b| table[usize::from(b)] != 0)
                            {
                                end += 1;
                            }
                            expected = Some((start, end));
                            break;
                        }
                    }
                    let found = pattern.find(subject, initial).unwrap();
                    assert_eq!(
                        found.as_ref().map(|found| (found.start(), found.end())),
                        expected,
                        "{source:?} {subject:?} {initial}"
                    );
                    if *captured {
                        assert_eq!(
                            found.map(|found| (found.captures[0].start(), found.captures[0].end())),
                            expected,
                            "{source:?} {subject:?} {initial}"
                        );
                    }
                }
            }
        }
        for source in [
            &b"%d*"[..],
            b"%d+x",
            b"(%d+)x",
            b"(%d*)",
            b"()%d+",
            b"%d-",
            b"%d?",
            b"%d+$",
            b"abc",
        ] {
            assert!(
                Pattern::new(source).unwrap().run_class().is_none(),
                "{source:?}"
            );
        }
    }

    #[test]
    fn byte_tables_match_the_class_and_set_parsers_for_every_byte() {
        for class in 0..=u8::MAX {
            let table = class_set(class);
            for byte in 0..=u8::MAX {
                assert_eq!(
                    table.contains(byte),
                    match_class(byte, class),
                    "%{class} {byte}"
                );
            }
        }
        let sets: &[&[u8]] = &[
            b"[a]",
            b"[^a]",
            b"[]]",
            b"[^]]",
            b"[]a]",
            b"[a-c]",
            b"[^a-c]",
            b"[a-]",
            b"[-a]",
            b"[%a-z]",
            b"[a-%d]",
            b"[z-a]",
            b"[%]]",
            b"[%w_%.]",
            b"[^%s%d]",
            b"[%Q%z%Z]",
            b"[\x00-\xff]",
            b"[^\x00-\xff]",
            b"[a-c-e]",
            b"[--/]",
            b"[%%-%]]",
            b"[%a%A]",
        ];
        for set in sets {
            let table = set_table(set, 0).unwrap();
            for byte in 0..=u8::MAX {
                assert_eq!(
                    table.contains(byte),
                    match_set(byte, set, 0).unwrap(),
                    "{} {byte}",
                    String::from_utf8_lossy(set)
                );
            }
        }
    }
}
