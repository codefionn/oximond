//! The pattern library calls of a running trace: `string.find`,
//! `string.match`, `string.gmatch` (the iterator and its steps) and
//! `string.gsub` ([`oximond::jit_ir::trace::HelperId::string_find_pattern`] and
//! the other helpers of its family).
//!
//! `find` and `match` run the native the interpreter runs, so their results,
//! their start positions and their errors are the interpreter's. The
//! recorder knows how many results the recorded call gave and which are
//! strings; the helper of that shape misses when a later call gives another
//! shape, and the interpreter replays the call. `gmatch` and `gsub` have
//! no native a helper could run (the iterator's step and the replacement are
//! the VM's own code), so the code of the interpreter's
//! `call_gmatch_iterator` and `try_string_gsub_direct` is repeated here for
//! the cases it answers without an error.
//!
//! A helper here raises no error: a malformed pattern, a bad capture index, a
//! replacement the profile's rules reject and every argument of another type
//! make it miss, before it changes anything.

use super::{
    HostState, MAX_HOST_STRING_BYTES, claim_lane, lane_value, publish_string, text,
    with_values_apart,
};
use crate::jit::{
    TRACE_TAG_BORROWED_STRING, TRACE_TAG_CLOSURE, TRACE_TAG_FLOAT, TRACE_TAG_INTEGER,
    TRACE_TAG_NIL, TRACE_TAG_STRING, TRACE_TAG_TABLE, TraceInputSlot,
};
use crate::pattern::{Capture, GsubPlan, Match, MatchRef, Pattern, Template};
use crate::runtime::{
    DirectNativeKind, GmatchIteratorHandle, NativeFunctionHandle, NativePlan, Runtime, Value,
    format_integer,
};
use crate::string::LuaString;
use crate::vm::continuations::{gsub_literal, next_gsub_search, skip_gsub_empty_match};

/// The results a helper gives: how many and which are strings (bit `i` set:
/// result `i` is a string, otherwise a position).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::vm::jit::trace) struct Shape {
    pub(in crate::vm::jit::trace) count: usize,
    pub(in crate::vm::jit::trace) strings: usize,
    /// Whether a match with more results than `count` has the shape too: the
    /// call wants the first `count` only. Without, the counts must be equal.
    pub(in crate::vm::jit::trace) prefix: bool,
}

impl Shape {
    /// The shape of the first `wanted` results of a match of this shape, which
    /// leads with `lead` positions (a `string.find` starts with two), as the
    /// shape of every match whose first results are those. `None` when the
    /// match has fewer results than `wanted`: the call fills those with nil,
    /// and only a match of exactly this shape does the same.
    pub(in crate::vm::jit::trace) fn wanting(self, wanted: usize, lead: usize) -> Option<Self> {
        let count = wanted.max(lead + usize::from(lead == 0));
        if count > self.count {
            return None;
        }
        let captures = count - lead;
        Some(Self {
            count,
            strings: self.strings & ((1_usize << captures) - 1),
            prefix: true,
        })
    }
}

/// The shape of the results of a match: its captures, or the whole match.
pub(in crate::vm::jit::trace) fn shape_of(found: &Match) -> Shape {
    let captures = found.captures();
    let mut strings = 0;
    for (position, capture) in captures.iter().enumerate() {
        if !capture.is_position() {
            strings |= 1 << position;
        }
    }
    if captures.is_empty() {
        return Shape {
            count: 1,
            strings: 1,
            prefix: false,
        };
    }
    Shape {
        count: captures.len(),
        strings,
        prefix: false,
    }
}

/// Runs a native on `args` and takes all its results. `None` for an error, a
/// `Redispatch`, or a native the runtime alone cannot run.
fn call_all(
    runtime: &mut Runtime,
    handle: NativeFunctionHandle,
    args: &[Value],
) -> Option<Vec<Value>> {
    match runtime.native_plan(handle) {
        NativePlan::Scalar(callback)
        | NativePlan::Unary(callback, _)
        | NativePlan::Binary(callback, _) => callback(runtime, args).ok().map(|value| vec![value]),
        NativePlan::Paired(callback, _) | NativePlan::Multiple(callback) => {
            callback(runtime, args).ok()
        }
        NativePlan::Other => None,
    }
}

/// The word a result of the shape's kind stands for: the lane of a new
/// string, or the position.
fn publish_word(
    bank: &mut Vec<Value>,
    state: &mut HostState,
    lengths: &mut [u32],
    value: &Value,
    string: bool,
) -> Option<i64> {
    if string {
        let Value::String(text) = value else {
            return None;
        };
        let mut slot = TraceInputSlot::other();
        return publish_string(bank, state, lengths, text.clone(), &mut slot).then(|| slot.bits());
    }
    match value {
        Value::Integer(position) => Some(*position),
        #[allow(clippy::cast_possible_truncation, reason = "the round trip rejects")]
        Value::Number(position) => {
            let integer = *position as i64;
            #[allow(clippy::float_cmp, clippy::cast_precision_loss, reason = "exact")]
            (integer as f64 == *position).then_some(integer)
        }
        _ => None,
    }
}

/// `string.find` or `string.match` (the native of `kind`) of a call that
/// found a match and gave the results of `shape`, or none when `shape` is
/// `None`, which the call must then give as one nil. Writes the words of the
/// results to `words`. `prepared` is the pattern a trace validated when it
/// compiled, for a call whose pattern is a constant.
#[allow(
    clippy::too_many_arguments,
    reason = "the run's pieces, the call and its answer"
)]
pub(in crate::vm::jit::trace) fn through_native_all(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    kept: (&mut HostState, &mut [u32]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    shape: Option<Shape>,
    words: &mut [i64],
    prepared: Option<&Pattern>,
) -> bool {
    through_native_all_noting(
        runtime, bank, roots, kept, kind, args, shape, words, prepared, &mut false,
    )
}

/// [`through_native_all`] that sets `used` when the `prepared` pattern gave
/// the answer, and not the native it falls back to.
#[allow(
    clippy::too_many_arguments,
    reason = "the run's pieces, the call and its answer"
)]
pub(in crate::vm::jit::trace) fn through_native_all_noting(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    shape: Option<Shape>,
    words: &mut [i64],
    prepared: Option<&Pattern>,
    used: &mut bool,
) -> bool {
    if let Some(answer) = through_pattern(
        runtime,
        bank,
        roots,
        (state, lengths),
        kind,
        args,
        shape,
        words,
        prepared,
    ) {
        *used = prepared.is_some();
        return answer;
    }
    let Some(handle) = runtime.direct_native(kind) else {
        return false;
    };
    let Some(values) = with_values_apart(args, bank, roots, |values| {
        call_all(runtime, handle, values)
    })
    .flatten() else {
        return false;
    };
    let Some(shape) = shape else {
        return matches!(values.as_slice(), [Value::Nil]);
    };
    // `find` leads with the start and the end of the match, which are
    // positions.
    let lead = if kind == DirectNativeKind::Find { 2 } else { 0 };
    let fits = if shape.prefix {
        values.len() >= shape.count
    } else {
        values.len() == shape.count
    };
    if !fits || shape.count > words.len() {
        return false;
    }
    for (position, value) in values.iter().enumerate().take(shape.count) {
        let string = position >= lead && shape.strings & (1 << (position - lead)) != 0;
        match publish_word(bank, state, lengths, value, string) {
            Some(word) => words[position] = word,
            None => return false,
        }
    }
    true
}

/// Where the search of a call of `string.find` or `string.match` starts.
enum Search {
    /// The call finds nothing without searching (a position past the end, on
    /// the profiles that fail it).
    Nothing,
    /// At this byte.
    From(usize),
}

/// The search that a call of `string.find` or `string.match` with the initial
/// position `initial` makes in a subject of `length` bytes.
fn search_start(runtime: &Runtime, length: usize, initial: isize) -> Option<Search> {
    let length = isize::try_from(length).ok()?;
    let mut initial = if initial < 0 {
        length.saturating_add(initial).saturating_add(1)
    } else {
        initial
    };
    if initial > length.saturating_add(1) {
        // Lua 5.1 and LuaJIT start a search past the end at the end; 5.3 and
        // later fail it (see `string_find` and `string_match`).
        if !runtime.string_init_clamps() {
            return Some(Search::Nothing);
        }
        initial = length.saturating_add(1);
    }
    Some(Search::From(usize::try_from(initial.max(1) - 1).ok()?))
}

/// The initial position a slot holds, as [`small_index`] takes one.
fn slot_index(slot: Option<&TraceInputSlot>) -> Option<isize> {
    match slot {
        None => Some(1),
        Some(slot) => match slot.tag() {
            TRACE_TAG_NIL => Some(1),
            TRACE_TAG_INTEGER => small_index(Some(&Value::Integer(slot.bits())), 1),
            TRACE_TAG_FLOAT => small_index(
                Some(&Value::Number(f64::from_bits(slot.bits().cast_unsigned()))),
                1,
            ),
            _ => None,
        },
    }
}

/// `string.find` or `string.match` run on the pattern directly, as the
/// interpreter's in-place path does, without the native's result vector.
/// `None` for a call it leaves to the native (plain text, an argument of
/// another type); `Some(false)` is a miss.
#[allow(
    clippy::too_many_arguments,
    reason = "the run's pieces, the call and its answer"
)]
fn through_pattern(
    runtime: &Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    kind: DirectNativeKind,
    args: &[TraceInputSlot],
    shape: Option<Shape>,
    words: &mut [i64],
    prepared: Option<&Pattern>,
) -> Option<bool> {
    let find = kind == DirectNativeKind::Find;
    if !(2..=if find { 4 } else { 3 }).contains(&args.len()) {
        return None;
    }
    // A plain search is the native's.
    if find
        && args
            .get(3)
            .is_some_and(|slot| !matches!(slot.tag(), TRACE_TAG_NIL))
    {
        return None;
    }
    let initial = slot_index(args.get(2))?;
    let source = text(args[0], bank, roots)?.clone();
    let shared;
    let pattern = if let Some(prepared) = prepared {
        prepared
    } else {
        let pattern = text(args[1], bank, roots)?;
        let Ok(found) = Pattern::shared(pattern.as_bytes(), false) else {
            // A malformed pattern raises in the interpreter.
            return Some(false);
        };
        shared = found;
        &*shared
    };
    let bytes = source.as_bytes();
    let Search::From(start) = search_start(runtime, bytes.len(), initial)? else {
        return Some(shape.is_none());
    };
    let lead = if find { 2 } else { 0 };
    let answered = pattern.find_with(bytes, start, |found| {
        // A match where the recording found none is a miss.
        shape.is_some_and(|shape| {
            match_words(bank, (state, lengths), bytes, found, (shape, lead), words)
        })
    });
    Some(match answered {
        Ok(Some(hit)) => hit,
        Ok(None) => shape.is_none(),
        Err(_) => false,
    })
}

/// What the helpers read of a match: [`Match`] or the matcher's own
/// [`MatchRef`].
pub(in crate::vm::jit::trace) trait Found {
    fn start(&self) -> usize;
    fn end(&self) -> usize;
    fn capture_count(&self) -> usize;
    fn capture(&self, index: usize) -> Capture;
}

impl Found for Match {
    fn start(&self) -> usize {
        Match::start(self)
    }
    fn end(&self) -> usize {
        Match::end(self)
    }
    fn capture_count(&self) -> usize {
        self.captures().len()
    }
    fn capture(&self, index: usize) -> Capture {
        self.captures()[index]
    }
}

impl Found for MatchRef<'_> {
    fn start(&self) -> usize {
        MatchRef::start(self)
    }
    fn end(&self) -> usize {
        MatchRef::end(self)
    }
    fn capture_count(&self) -> usize {
        MatchRef::capture_count(self)
    }
    fn capture(&self, index: usize) -> Capture {
        MatchRef::capture(self, index)
    }
}

/// The results of a match as the words of `shape`: the `lead` positions a
/// `string.find` starts with (the match's start and end), then its captures,
/// or for a match without captures (`lead` 0) the whole match. `None` when the
/// match gives another shape or the run is out of room for strings.
fn match_words(
    bank: &mut Vec<Value>,
    (state, lengths): (&mut HostState, &mut [u32]),
    source: &[u8],
    found: &impl Found,
    (shape, lead): (Shape, usize),
    words: &mut [i64],
) -> bool {
    let captures = found.capture_count();
    let count = lead + if lead == 0 { captures.max(1) } else { captures };
    let fits = if shape.prefix {
        count >= shape.count
    } else {
        count == shape.count
    };
    if !fits || shape.count > words.len() {
        return false;
    }
    let mut string = |position: usize, bytes: &[u8], words: &mut [i64]| -> bool {
        let mut slot = TraceInputSlot::other();
        if !publish_string(bank, state, lengths, LuaString::from(bytes), &mut slot) {
            return false;
        }
        words[position] = slot.bits();
        true
    };
    if lead > 0 {
        words[0] = i64::try_from(found.start() + 1).unwrap_or(i64::MAX);
        words[1] = i64::try_from(found.end()).unwrap_or(i64::MAX);
    }
    if captures == 0 {
        return lead > 0
            || (shape.strings == 1 && string(0, &source[found.start()..found.end()], words));
    }
    for position in 0..captures.min(shape.count - lead) {
        let capture = found.capture(position);
        let wanted = shape.strings & (1 << position) != 0;
        if wanted == capture.is_position() {
            return false;
        }
        if wanted {
            if !string(
                lead + position,
                &source[capture.start()..capture.end()],
                words,
            ) {
                return false;
            }
        } else {
            words[lead + position] = i64::try_from(capture.start() + 1).unwrap_or(i64::MAX);
        }
    }
    true
}

/// `string.gmatch (s, p [, init])`: the native makes the iterator, which
/// takes a lane of the bank. Writes the lane to `out`.
pub(in crate::vm::jit::trace) fn gmatch_new(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    out: &mut i64,
) -> bool {
    if !(2..=3).contains(&args.len())
        || text(args[0], bank, roots).is_none()
        || text(args[1], bank, roots).is_none()
        || args.get(2).is_some_and(|slot| {
            !matches!(
                slot.tag(),
                TRACE_TAG_NIL | TRACE_TAG_INTEGER | TRACE_TAG_FLOAT
            )
        })
    {
        return false;
    }
    let Some(handle) = runtime.direct_native(DirectNativeKind::Gmatch) else {
        return false;
    };
    let Some(Some(mut values)) = with_values_apart(args, bank, roots, |values| {
        call_all(runtime, handle, values)
    }) else {
        return false;
    };
    let (Some(iterator @ Value::GmatchIterator(_)), true) = (values.pop(), values.is_empty())
    else {
        return false;
    };
    // The iterator weighs what its state does: a little.
    let Some(lane) = claim_lane(bank, state, lengths, 0, 32) else {
        return false;
    };
    if lane == bank.len() {
        bank.push(iterator);
    } else {
        bank[lane] = iterator;
    }
    *out = i64::try_from(lane).unwrap_or(0);
    true
}

/// What a step of an iterator found, and where its cursor goes.
pub(in crate::vm::jit::trace) struct Step {
    pub(in crate::vm::jit::trace) found: Option<Match>,
}

#[derive(Clone, Copy)]
enum Cursor {
    /// The iteration is over.
    End,
    /// The next search starts here; with the Lua 5.3 rules the match is
    /// remembered, and an empty match ending there is skipped.
    Search { at: usize, last_match: bool },
}

/// The step the interpreter's `call_gmatch_iterator` makes for `handle`,
/// without committing the cursor. `None` for a state or a pattern the
/// interpreter answers differently (an error, an iterator of `utf8.codes`).
pub(in crate::vm::jit::trace) fn gmatch_probe(
    runtime: &Runtime,
    handle: GmatchIteratorHandle,
) -> Option<(Step, LuaString)> {
    let step = gmatch_step_of(runtime, handle)?;
    let (source, ..) = runtime.gmatch_iterator_view(handle)?;
    Some((step, source.clone()))
}

/// [`gmatch_probe`] without the copy of the source, which stays in the
/// iterator.
fn gmatch_step_of(runtime: &Runtime, handle: GmatchIteratorHandle) -> Option<Step> {
    let (source, pattern, search, last_match, utf8_codes) = runtime.gmatch_iterator_view(handle)?;
    if utf8_codes {
        return None;
    }
    let bytes = source.as_bytes();
    if search > bytes.len() {
        return Some(Step { found: None });
    }
    let pattern = Pattern::shared(pattern.as_bytes(), true).ok()?;
    let modern = runtime.tagged_integers();
    let mut search = search;
    let found = loop {
        let found = pattern.find(bytes, search).ok()?;
        match found {
            Some(found) if skip_gsub_empty_match(modern, &found, last_match) => {
                if found.end() == bytes.len() {
                    break None;
                }
                search = found.end() + 1;
            }
            found => break found,
        }
    };
    Some(Step { found })
}

/// How many more steps than the one just made the iterator `handle` gives
/// before it ends, counted up to `limit` (a loop that runs `limit` more
/// trips is as long as one that runs more). `None` for an iterator whose
/// steps the helpers do not make. Nothing is committed: the cursor is
/// followed on a copy.
pub(in crate::vm::jit::trace) fn gmatch_remaining(
    runtime: &Runtime,
    handle: GmatchIteratorHandle,
    limit: usize,
) -> Option<usize> {
    let (source, pattern, mut search, mut last_match, utf8_codes) =
        runtime.gmatch_iterator_view(handle)?;
    if utf8_codes {
        return None;
    }
    let bytes = source.as_bytes();
    let pattern = Pattern::shared(pattern.as_bytes(), true).ok()?;
    let modern = runtime.tagged_integers();
    let mut steps = 0;
    while steps < limit && search <= bytes.len() {
        let found = loop {
            let found = pattern.find_with(bytes, search, |found| (found.start(), found.end()));
            match found.ok()? {
                Some((start, end)) if modern && start == end && last_match == Some(end) => {
                    if end == bytes.len() {
                        break None;
                    }
                    search = end + 1;
                }
                found => break found,
            }
        };
        let Some((start, end)) = found else {
            break;
        };
        steps += 1;
        if modern {
            search = end;
            last_match = Some(end);
        } else {
            search = if start == end {
                if end == bytes.len() {
                    bytes.len().saturating_add(1)
                } else {
                    end + 1
                }
            } else {
                end
            };
        }
    }
    Some(steps)
}

/// The iterator in the bank lane `slot` names.
fn iterator_of(slot: TraceInputSlot, bank: &[Value]) -> Option<GmatchIteratorHandle> {
    if slot.tag() != TRACE_TAG_CLOSURE {
        return None;
    }
    match bank.get(usize::try_from(slot.bits()).ok()?)? {
        Value::GmatchIterator(handle) => Some(*handle),
        _ => None,
    }
}

/// A step of a `string.gmatch` iterator. With a `shape`, the step must give
/// a match of that shape; without, it must find none. Either way it commits
/// the cursor, and only after everything that could miss. The match is read
/// where the matcher holds it ([`Pattern::find_with`]), as
/// [`gmatch_step_of`] reads it, so a step copies nothing it does not publish.
/// `prepared` is the pattern a trace prepared for the iterator when it
/// compiled; an iterator of another pattern takes the shared one.
pub(in crate::vm::jit::trace) fn gmatch_step(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    kept: (&mut HostState, &mut [u32]),
    iterator: TraceInputSlot,
    shape: Option<Shape>,
    words: &mut [i64],
    prepared: Option<&Pattern>,
) -> bool {
    gmatch_step_noting(
        runtime, bank, kept, iterator, shape, words, prepared, &mut false,
    )
}

/// [`gmatch_step`] that sets `used` when the step read the `prepared` pattern
/// (the iterator was made from it) and not the shared one.
#[allow(clippy::too_many_arguments)]
pub(in crate::vm::jit::trace) fn gmatch_step_noting(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    kept: (&mut HostState, &mut [u32]),
    iterator: TraceInputSlot,
    shape: Option<Shape>,
    words: &mut [i64],
    prepared: Option<&Pattern>,
    used: &mut bool,
) -> bool {
    let Some(handle) = iterator_of(iterator, bank) else {
        return false;
    };
    let (state, lengths) = kept;
    let modern = runtime.tagged_integers();
    let Some((source, pattern, mut search, last_match, utf8_codes)) =
        runtime.gmatch_iterator_view(handle)
    else {
        return false;
    };
    if utf8_codes {
        return false;
    }
    let bytes = source.as_bytes();
    let length = bytes.len();
    let cursor = if search > length {
        if shape.is_some() {
            return false;
        }
        Cursor::End
    } else {
        let shared;
        let pattern = if let Some(prepared) =
            prepared.filter(|prepared| prepared.source() == pattern.as_bytes())
        {
            *used = true;
            prepared
        } else {
            let Ok(found) = Pattern::shared(pattern.as_bytes(), true) else {
                return false;
            };
            shared = found;
            &*shared
        };
        loop {
            let found = pattern.find_with(bytes, search, |found| {
                let (start, end) = (found.start(), found.end());
                if modern && start == end && last_match == Some(end) {
                    return Err(end);
                }
                // A match where the step was recorded to end, or the other
                // way round, is a miss.
                Ok((
                    start,
                    end,
                    shape.is_some_and(|shape| {
                        match_words(
                            bank,
                            (&mut *state, &mut *lengths),
                            bytes,
                            found,
                            (shape, 0),
                            words,
                        )
                    }),
                ))
            });
            match found {
                Err(_) => return false,
                Ok(Some(Err(end))) => {
                    if end == length {
                        break Cursor::End;
                    }
                    search = end + 1;
                }
                Ok(Some(Ok((start, end, published)))) => {
                    if !published {
                        return false;
                    }
                    break if modern {
                        Cursor::Search {
                            at: end,
                            last_match: true,
                        }
                    } else {
                        Cursor::Search {
                            at: if start == end {
                                if end == length {
                                    length.saturating_add(1)
                                } else {
                                    end + 1
                                }
                            } else {
                                end
                            },
                            last_match: false,
                        }
                    };
                }
                Ok(None) => break Cursor::End,
            }
        }
    };
    if matches!(cursor, Cursor::End) && shape.is_some() {
        return false;
    }
    let committed = match cursor {
        Cursor::End => runtime.set_gmatch_iterator_search(handle, length.saturating_add(1)),
        Cursor::Search {
            at,
            last_match: true,
        } => runtime.set_gmatch_iterator_last_match(handle, at),
        Cursor::Search { at, .. } => runtime.set_gmatch_iterator_search(handle, at),
    };
    committed.is_ok()
}

/// What `string.gsub` replaces a match with.
pub(in crate::vm::jit::trace) enum Replacement<'a> {
    Template(&'a [u8]),
    /// A template parsed when the trace compiled.
    Prepared(&'a Template),
    Table(crate::runtime::TableHandle),
}

/// The limit on the replacements of `string.gsub` that the live value `limit`
/// gives, as `gsub_limit` takes it.
pub(in crate::vm::jit::trace) fn live_limit(limit: Option<&Value>) -> Option<usize> {
    match limit {
        None | Some(Value::Nil) => Some(usize::MAX),
        Some(Value::Integer(count)) => Some(usize::try_from(*count).unwrap_or(0)),
        Some(Value::Number(count)) if !count.is_finite() => None,
        Some(Value::Number(count)) if *count <= 0.0 => Some(0),
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(Value::Number(count)) => Some(count.trunc() as usize),
        _ => None,
    }
}

/// Appends the template `template` for the match `found` of `source`, as
/// `gsub_template` does. `legacy` is the Lua 5.1 family, which copies the
/// character after a `%` and a NUL for a final `%`. `false` for a template
/// the profile rejects (a `%` before another character or at the end on the
/// others, a capture the pattern lacks).
fn expand_template(
    output: &mut Vec<u8>,
    template: &[u8],
    source: &[u8],
    found: &Match,
    legacy: bool,
) -> bool {
    let mut index = 0;
    while index < template.len() {
        if template[index] != b'%' {
            let end = template[index..]
                .iter()
                .position(|byte| *byte == b'%')
                .map_or(template.len(), |at| index + at);
            output.extend_from_slice(&template[index..end]);
            index = end;
            continue;
        }
        let Some(&next) = template.get(index + 1) else {
            if legacy {
                output.push(0);
                break;
            }
            return false;
        };
        match next {
            b'%' => output.push(b'%'),
            b'0' => output.extend_from_slice(&source[found.start()..found.end()]),
            b'1'..=b'9' => {
                if found.captures().is_empty() && next == b'1' {
                    output.extend_from_slice(&source[found.start()..found.end()]);
                } else {
                    let Some(capture) = found.captures().get(usize::from(next - b'1')) else {
                        return false;
                    };
                    if capture.is_position() {
                        output.extend_from_slice((capture.start() + 1).to_string().as_bytes());
                    } else {
                        output.extend_from_slice(&source[capture.start()..capture.end()]);
                    }
                }
            }
            _ if legacy => output.push(next),
            _ => return false,
        }
        index += 2;
    }
    true
}

/// The result of `string.gsub (source, pattern, replacement, maximum)`: the
/// new string's bytes and the number of replacements. `None` for a call the
/// interpreter might raise an error for, or answer through Lua code (a table
/// with an `__index`).
#[allow(clippy::too_many_lines)]
pub(in crate::vm::jit::trace) fn gsub_compute(
    runtime: &Runtime,
    bytes: &[u8],
    pattern: &[u8],
    replacement: &Replacement<'_>,
    maximum: usize,
) -> Option<(Vec<u8>, usize)> {
    let pattern = Pattern::shared(pattern, false).ok()?;
    gsub_with(runtime, bytes, &pattern, replacement, maximum)
}

/// [`gsub_compute`] with the pattern already built.
#[allow(clippy::too_many_lines)]
pub(in crate::vm::jit::trace) fn gsub_with(
    runtime: &Runtime,
    bytes: &[u8],
    pattern: &Pattern,
    replacement: &Replacement<'_>,
    maximum: usize,
) -> Option<(Vec<u8>, usize)> {
    if let Replacement::Table(table) = replacement {
        // A table with an `__index` may run Lua code.
        match runtime.raw_metatable(&Value::Table(*table)) {
            Ok(None) => {}
            Ok(Some(metatable))
                if runtime
                    .table_get_string_ref(metatable, &LuaString::from("__index"))
                    .is_none() => {}
            _ => return None,
        }
    }
    let modern = runtime.tagged_integers();
    let plain = match replacement {
        Replacement::Template(template) => (!template.contains(&b'%')).then_some(*template),
        Replacement::Prepared(template) => template.is_plain().then(|| template.bytes()),
        Replacement::Table(_) => None,
    };
    if let Some(template) = plain {
        return gsub_literal(
            bytes,
            pattern,
            template,
            (maximum, modern, MAX_HOST_STRING_BYTES),
        );
    }
    let length = bytes.len();
    let anchored = pattern.is_anchored();
    let mut output = Vec::with_capacity(length);
    let mut cursor = 0;
    let mut search = 0;
    let mut count = 0_usize;
    let mut last_match = None;
    while count < maximum && search <= length {
        let found = pattern.find(bytes, search).ok()?;
        let Some(found) = found else {
            break;
        };
        if skip_gsub_empty_match(modern, &found, last_match) {
            search = found.end().saturating_add(1);
            continue;
        }
        output.extend_from_slice(&bytes[cursor..found.start()]);
        match replacement {
            Replacement::Template(template) => {
                if !expand_template(
                    &mut output,
                    template,
                    bytes,
                    &found,
                    runtime.legacy_pattern_errors(),
                ) {
                    return None;
                }
            }
            Replacement::Prepared(template) => {
                if !template.expand(&mut output, bytes, &found) {
                    return None;
                }
            }
            Replacement::Table(table) => {
                let key = match found.captures().first() {
                    Some(capture) if capture.is_position() => {
                        runtime.integer_value(i64::try_from(capture.start() + 1).ok()?)
                    }
                    Some(capture) => Value::from(bytes[capture.start()..capture.end()].to_vec()),
                    None => Value::from(bytes[found.start()..found.end()].to_vec()),
                };
                match runtime.table_get(*table, &key) {
                    Ok(Value::Nil | Value::Boolean(false)) => {
                        output.extend_from_slice(&bytes[found.start()..found.end()]);
                    }
                    Ok(Value::String(value)) => output.extend_from_slice(value.as_bytes()),
                    Ok(Value::Integer(value)) => output.extend_from_slice(&format_integer(value)),
                    Ok(Value::Number(value)) => {
                        output.extend_from_slice(&runtime.number_format().format(value));
                    }
                    _ => return None,
                }
            }
        }
        if output.len() > MAX_HOST_STRING_BYTES {
            return None;
        }
        cursor = found.end();
        last_match = Some(found.end());
        search = if anchored {
            length.saturating_add(1)
        } else {
            next_gsub_search(modern, &found, length)
        };
        count += 1;
    }
    output.extend_from_slice(&bytes[cursor..]);
    Some((output, count))
}

/// `string.gsub (s, p, r [, n])` with a string or a table `r`. Writes the
/// new string's lane and the count of replacements to `words`.
pub(in crate::vm::jit::trace) fn gsub(
    runtime: &mut Runtime,
    bank: &mut Vec<Value>,
    roots: &[Value],
    (state, lengths): (&mut HostState, &mut [u32]),
    args: &[TraceInputSlot],
    words: &mut [i64],
    prepared: Option<&GsubPlan>,
) -> bool {
    if !(3..=4).contains(&args.len()) {
        return false;
    }
    let (Some(source), Some(pattern)) = (text(args[0], bank, roots), text(args[1], bank, roots))
    else {
        return false;
    };
    let limit = match args.get(3) {
        None => Value::Nil,
        Some(slot) => match slot.tag() {
            TRACE_TAG_NIL => Value::Nil,
            TRACE_TAG_INTEGER => Value::Integer(slot.bits()),
            TRACE_TAG_FLOAT => Value::Number(f64::from_bits(slot.bits().cast_unsigned())),
            _ => return false,
        },
    };
    let Some(maximum) = live_limit(Some(&limit)) else {
        return false;
    };
    // The strings stay where they are until the new one is made: the result
    // is the only thing the bank takes.
    let replacement = match args[2].tag() {
        TRACE_TAG_STRING | TRACE_TAG_BORROWED_STRING => {
            match (
                prepared.and_then(|plan| plan.template.as_ref()),
                text(args[2], bank, roots),
            ) {
                (Some(template), Some(_)) => Replacement::Prepared(template),
                (None, Some(template)) => Replacement::Template(template.as_bytes()),
                _ => return false,
            }
        }
        TRACE_TAG_TABLE => match lane_value(args[2], bank, roots) {
            Some(Value::Table(table)) => Replacement::Table(*table),
            _ => return false,
        },
        _ => return false,
    };
    let computed = match prepared {
        Some(plan) => gsub_with(
            runtime,
            source.as_bytes(),
            &plan.pattern,
            &replacement,
            maximum,
        ),
        None => gsub_compute(
            runtime,
            source.as_bytes(),
            pattern.as_bytes(),
            &replacement,
            maximum,
        ),
    };
    let Some((output, count)) = computed else {
        return false;
    };
    let mut slot = TraceInputSlot::other();
    if !publish_string(bank, state, lengths, LuaString::from(output), &mut slot) {
        return false;
    }
    words[0] = slot.bits();
    words[1] = i64::try_from(count).unwrap_or(i64::MAX);
    true
}

/// A small string index (a position argument of the library): nil is
/// `default`, a number is taken when it is integral and fits 32 bits, as the
/// natives take theirs without a conversion. `None` for any other value.
fn small_index(argument: Option<&Value>, default: isize) -> Option<isize> {
    match argument {
        None | Some(Value::Nil) => Some(default),
        Some(Value::Integer(integer)) => i32::try_from(*integer).ok().map(|small| small as isize),
        Some(Value::Number(number)) => {
            #[allow(clippy::cast_possible_truncation, reason = "the round trip rejects")]
            let small = *number as i32;
            #[allow(clippy::float_cmp, reason = "an exact round trip is the test")]
            (f64::from(small) == *number).then_some(small as isize)
        }
        _ => None,
    }
}

/// How a `string.find` or `string.match` ends, as the recorder predicts it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::vm::jit::trace) enum Outcome {
    /// A match with the results of the shape.
    Found(Shape),
    /// No match: the call gives one nil.
    Absent,
}

/// What `string.find` and `string.match` give for the live arguments `live`,
/// as the recorder predicts it, and `None` for a call the recorder leaves to
/// the interpreter (an error, an argument of an unusual type). A match
/// may have more results than a helper gives: the recorder takes the ones its
/// call wants (see [`Shape::wanting`]).
pub(in crate::vm::jit::trace) fn predict(
    runtime: &Runtime,
    kind: DirectNativeKind,
    live: &[Value],
) -> Option<Outcome> {
    let (Value::String(source), Value::String(pattern)) = (live.first()?, live.get(1)?) else {
        return None;
    };
    let find = kind == DirectNativeKind::Find;
    let initial = small_index(live.get(2), 1)?;
    let plain = match live.get(3) {
        _ if !find => false,
        None | Some(Value::Nil | Value::Boolean(false)) => false,
        Some(Value::Boolean(true) | Value::Integer(_) | Value::Number(_)) => true,
        Some(_) => return None,
    };
    let bytes = source.as_bytes();
    let Search::From(start) = search_start(runtime, bytes.len(), initial)? else {
        return Some(Outcome::Absent);
    };
    if plain {
        let needle = pattern.as_bytes();
        let found = needle.is_empty()
            || bytes
                .get(start..)?
                .windows(needle.len())
                .any(|window| window == needle);
        return Some(if found {
            Outcome::Found(Shape {
                count: 2,
                strings: 0,
                prefix: false,
            })
        } else {
            Outcome::Absent
        });
    }
    let pattern = Pattern::shared(pattern.as_bytes(), false).ok()?;
    let Some(found) = pattern.find(bytes, start).ok()? else {
        return Some(Outcome::Absent);
    };
    if find {
        let mut shape = shape_of(&found);
        if found.captures().is_empty() {
            shape = Shape {
                count: 0,
                strings: 0,
                prefix: false,
            };
        }
        shape.count += 2;
        return Some(Outcome::Found(shape));
    }
    Some(Outcome::Found(shape_of(&found)))
}
