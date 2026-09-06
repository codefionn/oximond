//! The library helpers of the VM ([`oximond::jit_ir::trace::TraceOp::CallHelper`]):
//! `string.char`, `tostring`, `string.format`, `..`, `string.find`,
//! `table.concat`, `getmetatable`, `setmetatable`, `table.insert`,
//! `table.remove`, `string.sub`, `string.rep`, `string.upper`,
//! `string.lower`, `string.reverse`, `math.random`, `math.modf`, `tonumber`
//! `string.byte`, `string.match`, `string.gmatch`, `string.gsub` and the check
//! before a method call on a string.
//!
//! Native code calls these `extern "C"` functions directly. Each takes the
//! run's [`RunTables`] as its context, the result block and the arguments as
//! entry tags and bits in registers (see the calling convention in
//! [`oximond::jit_ir::trace`]'s helper module), and returns 0 for an answer or
//! 1 for a miss. The work is the string library's own code in
//! [`super::host`]: the runtime's formatter, concatenation and table reads.
//!
//! A string answer is published as a new lane of the run's root bank by the
//! helper, under the run's room for strings (a lane count, a byte count and a
//! weight, see [`super::host`]). A run out of room is a miss: the trace
//! leaves before the call, the interpreter makes it and the garbage it makes
//! is the interpreter's. The helpers run no Lua code, raise no error and
//! collect nothing. They check everything before they publish, so a miss
//! changes nothing.

mod resume;
mod text;
mod vmcall;

use super::host::{self, pattern, pattern::Shape};
use super::native::RunTables;
use crate::jit::{
    TRACE_TAG_BOOLEAN, TRACE_TAG_BORROWED_STRING, TRACE_TAG_CLOSURE, TRACE_TAG_FLOAT,
    TRACE_TAG_INTEGER, TRACE_TAG_NIL, TRACE_TAG_STRING,
};
use crate::jit::{
    TRACE_TAG_TABLE, TraceInputSlot, TraceTables, install_helper, install_prepared_helper,
};
use crate::jit_ir::trace::{
    HelperId, MAX_PREFILLED_ARRAY, NEXT_BOX_ROW, NEXT_KINDS, NEXT_ROW, NextKind,
    PINNED_BUILTIN_STRING_FORMAT, PINNED_BUILTIN_STRING_GSUB, TABLE_NEW_PREFILL,
    pinned_builtin_identity, pinned_native_identity,
};
use crate::pattern::Pattern;
use crate::runtime::{
    BuiltinFunction, CellHandle, ClosureHandle, DirectNativeKind, Metamethod, NextKey, NextRef,
    OperatorHandler, Runtime, TableHandle, UserdataHandle, Value,
};
use crate::stdlib::PreparedFormat;
use crate::string::LuaString;
use crate::vm::arith::{
    call_metamethod_key, event, metamethod_key, newindex_metamethod_key, operator_event_key,
};

/// What a helper's answer is, which the slot the string library wrote must
/// match.
#[derive(Clone, Copy)]
enum Answer {
    /// A string lane.
    String,
    /// An integer (or a float that is one, as the profile has numbers).
    Integer,
    /// An integer, and nothing else.
    Whole,
    /// A float, and nothing else.
    Real,
    /// Nothing: the library must have answered nil.
    Nil,
}

/// The pieces of a run a library call needs.
type Call = fn(&mut RunTables<'static>, &[TraceInputSlot], &mut TraceInputSlot) -> bool;

/// Runs `call` on `args` and writes its answer to the result block `out`.
/// Returns the helper's status.
#[allow(unsafe_code)]
#[inline]
fn invoke(
    context: usize,
    out: *mut i64,
    answer: Answer,
    args: &[TraceInputSlot],
    call: Call,
) -> i64 {
    // SAFETY: `context` is the address of the `RunTables` that handed out the
    // host entry, which outlives the run and is used by nothing else while
    // native code calls here.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let mut slot = TraceInputSlot::other();
    if !call(tables, args, &mut slot) {
        return 1;
    }
    let word = match answer {
        Answer::String if slot.tag() == TRACE_TAG_STRING => slot.bits(),
        Answer::Integer | Answer::Whole if slot.tag() == TRACE_TAG_INTEGER => slot.bits(),
        Answer::Real if slot.tag() == TRACE_TAG_FLOAT => slot.bits(),
        Answer::Integer if slot.tag() == TRACE_TAG_FLOAT => {
            let number = f64::from_bits(slot.bits().cast_unsigned());
            #[allow(clippy::cast_possible_truncation, reason = "the round trip rejects")]
            let integer = number as i64;
            #[allow(
                clippy::float_cmp,
                clippy::cast_precision_loss,
                reason = "an exact round trip is the test"
            )]
            if integer as f64 != number {
                return 1;
            }
            integer
        }
        Answer::Nil if slot.tag() == TRACE_TAG_NIL => 0,
        _ => return 1,
    };
    if !matches!(answer, Answer::Nil) {
        // SAFETY: `out` is the control block's `helper_out`, which holds a
        // word per result and which nothing else touches during the call.
        unsafe { *out = word };
    }
    0
}

/// The library call `run` needs the parts of the run for, with the argument
/// count `count` out of the slots in `all`.
macro_rules! library_call {
    ($tables:ident, $args:ident, $out:ident => $body:expr) => {
        |$tables: &mut RunTables<'static>, $args: &[TraceInputSlot], $out: &mut TraceInputSlot| {
            $body
        }
    };
}

/// Defines the helper `$name`: `extern "C" fn(context, out, count, (tag,
/// bits) x N) -> status` for a variadic helper of N values.
macro_rules! variadic_helper {
    ($name:ident, $answer:expr, $call:expr, [$($tag:ident $bits:ident),+]) => {
        extern "C" fn $name(
            context: usize,
            out: *mut i64,
            count: i64,
            $($tag: i64, $bits: i64),+
        ) -> i64 {
            let all = [$(TraceInputSlot::from_words($tag, $bits)),+];
            let count = usize::try_from(count).unwrap_or(0).min(all.len());
            invoke(context, out, $answer, &all[..count], $call)
        }
    };
}

variadic_helper!(
    helper_string_char,
    Answer::String,
    library_call!(tables, args, out => {
        let (_, bank, _, state, lengths) = tables.library();
        host::string_char(bank, state, lengths, args, out)
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4]
);

variadic_helper!(
    helper_string_format,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::format_call(runtime, bank, roots, (state, lengths), args, out)
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4]
);

variadic_helper!(
    helper_string_format_luajit,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::through_native(
            runtime,
            bank,
            roots,
            (state, lengths),
            DirectNativeKind::StringFormatLuaJit,
            args,
            out,
        )
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4]
);

variadic_helper!(
    helper_string_format_native,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::through_native(
            runtime,
            bank,
            roots,
            (state, lengths),
            DirectNativeKind::Format,
            args,
            out,
        )
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4]
);

variadic_helper!(
    helper_concat,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::concat(runtime, bank, roots, (state, lengths), args, out, false)
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4]
);

variadic_helper!(
    helper_concat_error_text,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::concat(runtime, bank, roots, (state, lengths), args, out, true)
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4]
);

variadic_helper!(
    helper_string_find,
    Answer::Integer,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::find(runtime, bank, roots, (state, lengths), args, out)
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3]
);

variadic_helper!(
    helper_string_find_absent,
    Answer::Nil,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::find(runtime, bank, roots, (state, lengths), args, out)
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3]
);

variadic_helper!(
    helper_table_concat,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::table_concat(runtime, bank, roots, (state, lengths), args, out)
    }),
    [t0 b0, t1 b1, t2 b2, t3 b3]
);

/// `tostring` of a number or a string. A small number has its text ready in
/// the runtime.
#[allow(unsafe_code)]
extern "C" fn helper_tostring(context: usize, out: *mut i64, tag: i64, bits: i64) -> i64 {
    // The small number is the common call: answer it before the generic path
    // sets anything up.
    //
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let mut slot = TraceInputSlot::other();
    let (runtime, bank, _, state, lengths) = tables.library();
    if host::tostring_small(
        runtime,
        bank,
        state,
        lengths,
        TraceInputSlot::from_words(tag, bits),
        &mut slot,
    ) {
        // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
        unsafe { *out = slot.bits() };
        return 0;
    }
    invoke(
        context,
        out,
        Answer::String,
        &[TraceInputSlot::from_words(tag, bits)],
        library_call!(tables, args, out => {
            let (runtime, bank, roots, state, lengths) = tables.library();
            host::to_string(runtime, bank, roots, (state, lengths), args, out)
        }),
    )
}

/// Defines the helper `$name`: `extern "C" fn(context, out, (tag, bits) x N)
/// -> status` for a helper of N values that takes them all.
macro_rules! fixed_helper {
    ($name:ident, $answer:expr, $call:expr, [$($tag:ident $bits:ident),+]) => {
        extern "C" fn $name(context: usize, out: *mut i64, $($tag: i64, $bits: i64),+) -> i64 {
            let all = [$(TraceInputSlot::from_words($tag, $bits)),+];
            invoke(context, out, $answer, &all, $call)
        }
    };
}

variadic_helper!(
    helper_string_sub,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        if let Some(done) = host::string_sub(bank, roots, (state, lengths), args, out) {
            return done;
        }
        host::through_native(
            runtime,
            bank,
            roots,
            (state, lengths),
            DirectNativeKind::StringSlice,
            args,
            out,
        )
    }),
    [t0 b0, t1 b1, t2 b2]
);

variadic_helper!(
    helper_string_rep,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::string_rep(runtime, bank, roots, (state, lengths), args, out)
    }),
    [t0 b0, t1 b1, t2 b2]
);

fixed_helper!(
    helper_string_upper,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::through_native(
            runtime,
            bank,
            roots,
            (state, lengths),
            DirectNativeKind::StringUpper,
            args,
            out,
        )
    }),
    [t0 b0]
);

fixed_helper!(
    helper_string_lower,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::through_native(
            runtime,
            bank,
            roots,
            (state, lengths),
            DirectNativeKind::StringLower,
            args,
            out,
        )
    }),
    [t0 b0]
);

fixed_helper!(
    helper_string_reverse,
    Answer::String,
    library_call!(tables, args, out => {
        let (runtime, bank, roots, state, lengths) = tables.library();
        host::through_native(
            runtime,
            bank,
            roots,
            (state, lengths),
            DirectNativeKind::StringReverse,
            args,
            out,
        )
    }),
    [t0 b0]
);

macro_rules! tonumber_helper {
    ($name:ident, $answer:expr) => {
        variadic_helper!(
            $name,
            $answer,
            library_call!(tables, args, out => {
                let (runtime, bank, roots, state, lengths) = tables.library();
                host::tonumber(runtime, bank, roots, (state, lengths), args, out)
            }),
            [t0 b0, t1 b1]
        );
    };
}

tonumber_helper!(helper_tonumber_integer, Answer::Whole);
tonumber_helper!(helper_tonumber_float, Answer::Real);
tonumber_helper!(helper_tonumber_nil, Answer::Nil);

/// `string.byte` (see [`HelperId::STRING_BYTE`]): `wanted` bytes, then the
/// call's arguments.
#[allow(unsafe_code)]
extern "C" fn helper_string_byte(
    context: usize,
    out: *mut i64,
    count: i64,
    wanted: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
    ];
    let Some(given) = usize::try_from(count - 1).ok().filter(|given| *given <= 3) else {
        return 1;
    };
    let Some(wanted) = usize::try_from(wanted).ok().filter(|wanted| *wanted <= 4) else {
        return 1;
    };
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let mut bytes = [0_i64; 4];
    if !host::string_byte(runtime, bank, roots, &all[..given], &mut bytes[..wanted]) {
        return 1;
    }
    // SAFETY: `out` holds a word per result (see `invoke`).
    unsafe {
        for (position, byte) in bytes.iter().enumerate().take(wanted) {
            *out.add(position) = *byte;
        }
    }
    0
}

/// Copies the words a pattern helper made to the result block `out`.
#[allow(unsafe_code)]
fn write_words(out: *mut i64, words: &[i64]) {
    // SAFETY: `out` holds a word per result (see `invoke`).
    unsafe {
        for (position, word) in words.iter().enumerate() {
            *out.add(position) = *word;
        }
    }
}

/// `string.find` that found a match with `CAPTURES` captures, of which the
/// ones in the bits of `KINDS` are strings (see
/// [`HelperId::string_find_pattern`]).
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_find_pattern<const CAPTURES: usize, const KINDS: usize, const PREFIX: bool>(
    context: usize,
    out: *mut i64,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
    t3: i64,
    b3: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
        TraceInputSlot::from_words(t3, b3),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut words = [0_i64; 4];
    let shape = Shape {
        count: 2 + CAPTURES,
        strings: KINDS,
        prefix: PREFIX,
    };
    if !pattern::through_native_all(
        runtime,
        bank,
        roots,
        (state, lengths),
        DirectNativeKind::Find,
        &all[..count],
        Some(shape),
        &mut words,
        None,
    ) {
        return 1;
    }
    write_words(out, &words[..shape.count]);
    0
}

/// `string.match` that found a match and gave `RESULTS` results, of which
/// the ones in the bits of `KINDS` are strings (see
/// [`HelperId::string_match`]).
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_match<const RESULTS: usize, const KINDS: usize, const PREFIX: bool>(
    context: usize,
    out: *mut i64,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut words = [0_i64; 4];
    if !pattern::through_native_all(
        runtime,
        bank,
        roots,
        (state, lengths),
        DirectNativeKind::Match,
        &all[..count],
        Some(Shape {
            count: RESULTS,
            strings: KINDS,
            prefix: PREFIX,
        }),
        &mut words,
        None,
    ) {
        return 1;
    }
    write_words(out, &words[..RESULTS]);
    0
}

/// `string.match` that finds none.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_match_absent(
    context: usize,
    out: *mut i64,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let _ = out;
    i64::from(!pattern::through_native_all(
        runtime,
        bank,
        roots,
        (state, lengths),
        DirectNativeKind::Match,
        &all[..count],
        None,
        &mut [],
        None,
    ))
}

thread_local! {
    /// How many times a specialized pattern helper ran on this thread, for
    /// tests that check a constant pattern took the prepared path.
    static PREPARED_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many times a specialized helper ran on this thread.
pub(super) fn prepared_calls() -> u64 {
    PREPARED_CALLS.with(std::cell::Cell::get)
}

/// Counts a call of a specialized helper.
#[inline(always)]
fn count_prepared_call() {
    PREPARED_CALLS.with(|calls| calls.set(calls.get().wrapping_add(1)));
}

/// Defines the twin `$name` of a `string.format` helper for a constant
/// format: `prepared` is the address of the `PreparedFormat` the trace parsed
/// when it compiled, and `$native` the native the generic helper calls (`None`
/// for the modern form).
macro_rules! format_prepared_helper {
    ($name:ident, $native:expr) => {
        #[allow(unsafe_code, clippy::too_many_arguments)]
        extern "C" fn $name(
            context: usize,
            out: *mut i64,
            count: i64,
            prepared: usize,
            t0: i64,
            b0: i64,
            t1: i64,
            b1: i64,
            t2: i64,
            b2: i64,
            t3: i64,
            b3: i64,
            t4: i64,
            b4: i64,
        ) -> i64 {
            let all = [
                TraceInputSlot::from_words(t0, b0),
                TraceInputSlot::from_words(t1, b1),
                TraceInputSlot::from_words(t2, b2),
                TraceInputSlot::from_words(t3, b3),
                TraceInputSlot::from_words(t4, b4),
            ];
            let count = usize::try_from(count).unwrap_or(0).min(all.len());
            // SAFETY: see `invoke`.
            let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
            // SAFETY: the compiled code passes the address of a
            // `PreparedFormat` that its `TraceCode` keeps alive for as long
            // as the code can run.
            let plan = unsafe { &*(prepared as *const PreparedFormat) };
            let (runtime, bank, roots, state, lengths) = tables.library();
            let mut slot = TraceInputSlot::other();
            let mut used = false;
            if !host::format_prepared_call(
                runtime,
                bank,
                roots,
                (state, lengths),
                plan,
                $native,
                &all[..count],
                &mut slot,
                &mut used,
            ) || slot.tag() != TRACE_TAG_STRING
            {
                return 1;
            }
            if used {
                count_prepared_call();
            }
            // SAFETY: `out` is the control block's `helper_out`, which holds
            // a word per result and which nothing else touches during the
            // call.
            unsafe { *out = slot.bits() };
            0
        }
    };
}

format_prepared_helper!(helper_string_format_prepared, None);
format_prepared_helper!(
    helper_string_format_luajit_prepared,
    Some(DirectNativeKind::StringFormatLuaJit)
);
format_prepared_helper!(
    helper_string_format_native_prepared,
    Some(DirectNativeKind::Format)
);

/// [`helper_find_pattern`] for a constant pattern: `prepared` is the address
/// of the `Pattern` the trace validated when it compiled.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_find_pattern_prepared<
    const CAPTURES: usize,
    const KINDS: usize,
    const PREFIX: bool,
>(
    context: usize,
    out: *mut i64,
    count: i64,
    prepared: usize,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
    t3: i64,
    b3: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
        TraceInputSlot::from_words(t3, b3),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    // SAFETY: the compiled code passes the address of a `Pattern` that its
    // `TraceCode` keeps alive for as long as the code can run.
    let prepared = unsafe { &*(prepared as *const Pattern) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut words = [0_i64; 4];
    let shape = Shape {
        count: 2 + CAPTURES,
        strings: KINDS,
        prefix: PREFIX,
    };
    let mut used = false;
    if !pattern::through_native_all_noting(
        runtime,
        bank,
        roots,
        (state, lengths),
        DirectNativeKind::Find,
        &all[..count],
        Some(shape),
        &mut words,
        Some(prepared),
        &mut used,
    ) {
        return 1;
    }
    if used {
        count_prepared_call();
    }
    write_words(out, &words[..shape.count]);
    0
}

/// [`helper_match`] for a constant pattern (see
/// [`helper_find_pattern_prepared`]).
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_match_prepared<
    const RESULTS: usize,
    const KINDS: usize,
    const PREFIX: bool,
>(
    context: usize,
    out: *mut i64,
    count: i64,
    prepared: usize,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    // SAFETY: see `helper_find_pattern_prepared`.
    let prepared = unsafe { &*(prepared as *const Pattern) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut words = [0_i64; 4];
    let mut used = false;
    if !pattern::through_native_all_noting(
        runtime,
        bank,
        roots,
        (state, lengths),
        DirectNativeKind::Match,
        &all[..count],
        Some(Shape {
            count: RESULTS,
            strings: KINDS,
            prefix: PREFIX,
        }),
        &mut words,
        Some(prepared),
        &mut used,
    ) {
        return 1;
    }
    if used {
        count_prepared_call();
    }
    write_words(out, &words[..RESULTS]);
    0
}

/// [`helper_match_absent`] for a constant pattern (see
/// [`helper_find_pattern_prepared`]).
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_match_absent_prepared(
    context: usize,
    out: *mut i64,
    count: i64,
    prepared: usize,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
) -> i64 {
    prepared_absent(
        context,
        out,
        count,
        prepared,
        DirectNativeKind::Match,
        [t0, b0, t1, b1, t2, b2, 0, 0],
    )
}

/// [`helper_string_find_absent`] for a constant pattern with magic
/// characters (see [`helper_find_pattern_prepared`]).
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_find_absent_prepared(
    context: usize,
    out: *mut i64,
    count: i64,
    prepared: usize,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
    t3: i64,
    b3: i64,
) -> i64 {
    prepared_absent(
        context,
        out,
        count,
        prepared,
        DirectNativeKind::Find,
        [t0, b0, t1, b1, t2, b2, t3, b3],
    )
}

/// A call of a prepared pattern that finds nothing: status 0 when the call
/// gives the one nil.
#[allow(unsafe_code)]
#[inline(always)]
fn prepared_absent(
    context: usize,
    out: *mut i64,
    count: i64,
    prepared: usize,
    kind: DirectNativeKind,
    words: [i64; 8],
) -> i64 {
    let _ = out;
    let all = [
        TraceInputSlot::from_words(words[0], words[1]),
        TraceInputSlot::from_words(words[2], words[3]),
        TraceInputSlot::from_words(words[4], words[5]),
        TraceInputSlot::from_words(words[6], words[7]),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    // SAFETY: see `helper_find_pattern_prepared`.
    let prepared = unsafe { &*(prepared as *const Pattern) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut used = false;
    let answered = pattern::through_native_all_noting(
        runtime,
        bank,
        roots,
        (state, lengths),
        kind,
        &all[..count],
        None,
        &mut [],
        Some(prepared),
        &mut used,
    );
    if answered && used {
        count_prepared_call();
    }
    i64::from(!answered)
}

/// `string.gmatch (s, p [, init])`: the new iterator's lane.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_gmatch_new(
    context: usize,
    out: *mut i64,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut lane = 0;
    if !pattern::gmatch_new(
        runtime,
        bank,
        roots,
        (state, lengths),
        &all[..count],
        &mut lane,
    ) {
        return 1;
    }
    write_words(out, &[lane]);
    0
}

/// A step of a `string.gmatch` iterator that gives a match of `RESULTS`
/// results, shaped as `KINDS` says (see [`HelperId::gmatch_next`]).
#[allow(unsafe_code)]
extern "C" fn helper_gmatch_next<const RESULTS: usize, const KINDS: usize, const PREFIX: bool>(
    context: usize,
    out: *mut i64,
    lane: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, _, state, lengths) = tables.library();
    let mut words = [0_i64; 4];
    if !pattern::gmatch_step(
        runtime,
        bank,
        (state, lengths),
        TraceInputSlot::from_words(TRACE_TAG_CLOSURE, lane),
        Some(Shape {
            count: RESULTS,
            strings: KINDS,
            prefix: PREFIX,
        }),
        &mut words,
        None,
    ) {
        return 1;
    }
    write_words(out, &words[..RESULTS]);
    0
}

/// [`helper_gmatch_next`] for an iterator made from a constant pattern:
/// `prepared` is the address of the `Pattern` the trace validated when it
/// compiled. An iterator of another pattern takes the shared one.
#[allow(unsafe_code)]
extern "C" fn helper_gmatch_next_prepared<
    const RESULTS: usize,
    const KINDS: usize,
    const PREFIX: bool,
>(
    context: usize,
    out: *mut i64,
    prepared: usize,
    lane: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    // SAFETY: see `helper_find_pattern_prepared`.
    let prepared = unsafe { &*(prepared as *const Pattern) };
    let (runtime, bank, _, state, lengths) = tables.library();
    let mut words = [0_i64; 4];
    let mut used = false;
    if !pattern::gmatch_step_noting(
        runtime,
        bank,
        (state, lengths),
        TraceInputSlot::from_words(TRACE_TAG_CLOSURE, lane),
        Some(Shape {
            count: RESULTS,
            strings: KINDS,
            prefix: PREFIX,
        }),
        &mut words,
        Some(prepared),
        &mut used,
    ) {
        return 1;
    }
    if used {
        count_prepared_call();
    }
    write_words(out, &words[..RESULTS]);
    0
}

/// [`helper_gmatch_end`] for an iterator made from a constant pattern (see
/// [`helper_gmatch_next_prepared`]).
#[allow(unsafe_code)]
extern "C" fn helper_gmatch_end_prepared(
    context: usize,
    _out: *mut i64,
    prepared: usize,
    lane: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    // SAFETY: see `helper_find_pattern_prepared`.
    let prepared = unsafe { &*(prepared as *const Pattern) };
    let (runtime, bank, _, state, lengths) = tables.library();
    let mut used = false;
    let done = pattern::gmatch_step_noting(
        runtime,
        bank,
        (state, lengths),
        TraceInputSlot::from_words(TRACE_TAG_CLOSURE, lane),
        None,
        &mut [],
        Some(prepared),
        &mut used,
    );
    if done && used {
        count_prepared_call();
    }
    i64::from(!done)
}

/// The step of a `string.gmatch` iterator that ends the iteration.
extern "C" fn helper_gmatch_end(context: usize, _out: *mut i64, lane: i64) -> i64 {
    // SAFETY: see `invoke`.
    #[allow(unsafe_code)]
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, _, state, lengths) = tables.library();
    i64::from(!pattern::gmatch_step(
        runtime,
        bank,
        (state, lengths),
        TraceInputSlot::from_words(TRACE_TAG_CLOSURE, lane),
        None,
        &mut [],
        None,
    ))
}

/// `string.gsub (s, p, r [, n])` (see [`HelperId::STRING_GSUB`]).
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_gsub(
    context: usize,
    out: *mut i64,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
    t3: i64,
    b3: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
        TraceInputSlot::from_words(t3, b3),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut words = [0_i64; 2];
    if !pattern::gsub(
        runtime,
        bank,
        roots,
        (state, lengths),
        &all[..count],
        &mut words,
        None,
    ) {
        return 1;
    }
    write_words(out, &words);
    0
}

/// [`helper_gsub`] for a constant pattern: `prepared` is the address of the
/// `GsubPlan` the trace built when it compiled.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_gsub_prepared(
    context: usize,
    out: *mut i64,
    count: i64,
    prepared: usize,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
    t2: i64,
    b2: i64,
    t3: i64,
    b3: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
        TraceInputSlot::from_words(t2, b2),
        TraceInputSlot::from_words(t3, b3),
    ];
    let count = usize::try_from(count).unwrap_or(0).min(all.len());
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    // SAFETY: see `helper_find_pattern_prepared`, with a `GsubPlan`.
    let prepared = unsafe { &*(prepared as *const crate::pattern::GsubPlan) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut words = [0_i64; 2];
    if !pattern::gsub(
        runtime,
        bank,
        roots,
        (state, lengths),
        &all[..count],
        &mut words,
        Some(prepared),
    ) {
        return 1;
    }
    count_prepared_call();
    write_words(out, &words);
    0
}

/// The identity a trace pins for the function `value`, which a method call on
/// a string looks up: a native, or the builtin `string.format`.
pub(super) fn method_identity(value: &Value) -> Option<u64> {
    match value {
        Value::NativeFunction(function) => Some(pinned_native_identity(function.index())),
        Value::BuiltinFunction(BuiltinFunction::StringFormat) => {
            Some(pinned_builtin_identity(PINNED_BUILTIN_STRING_FORMAT))
        }
        Value::BuiltinFunction(BuiltinFunction::StringGsub) => {
            Some(pinned_builtin_identity(PINNED_BUILTIN_STRING_GSUB))
        }
        _ => None,
    }
}

/// The check before a method call on a string (see
/// [`HelperId::STRING_METHOD`]): the string metatable's `__index` is a table
/// that holds the pinned function under the key. Changes nothing.
#[allow(unsafe_code)]
extern "C" fn helper_string_method(
    context: usize,
    _out: *mut i64,
    key: i64,
    identity: i64,
    _receiver_tag: i64,
    _receiver: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let slot = TraceInputSlot::from_words(TRACE_TAG_STRING, key);
    let same = host::with_values(&[slot], bank, roots, |values| {
        let [Value::String(key)] = values else {
            return false;
        };
        // Nothing here makes a string: the runtime holds the shared
        // metatable and the `__index` key.
        let Some(metatable) = runtime.string_metatable() else {
            return false;
        };
        let Some(Value::Table(index)) =
            runtime.table_get_string_ref(metatable, runtime.index_metamethod_key())
        else {
            return false;
        };
        runtime
            .table_get_string_ref(*index, key)
            .and_then(method_identity)
            == Some(identity.cast_unsigned())
    });
    i64::from(same != Some(true))
}

/// The check before a call through `__call` (see [`HelperId::CALL_HANDLER`]):
/// the value in the slot is a table whose metatable's raw `__call` is the Lua
/// closure of the identity the trace pinned. The pin keeps that closure alive,
/// so no other closure can take its identity. Changes nothing.
#[allow(unsafe_code)]
extern "C" fn helper_call_handler(
    context: usize,
    _out: *mut i64,
    tag: i64,
    bits: i64,
    handler_identity: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let slot = TraceInputSlot::from_words(tag, bits);
    let same = host::with_values(&[slot], bank, roots, |values| {
        let [table @ Value::Table(_)] = values else {
            return false;
        };
        let Ok(Some(metatable)) = runtime.raw_metatable(table) else {
            return false;
        };
        matches!(
            runtime.table_get_string_ref(metatable, call_metamethod_key()),
            Some(Value::Closure(handler))
                if handler.native_identity() == handler_identity.cast_unsigned()
        )
    });
    i64::from(same != Some(true))
}

/// Where the walk of a metatable event (`__index` or `__newindex`) for a key
/// ends, as `Vm::run_index` makes it: while a table lacks the key, its
/// metatable's raw handler is a table, which the walk goes on in, or ends it.
pub(super) enum Walk {
    /// The table holds the key itself.
    Holds,
    /// The table lacks the key and its metatable has no handler for the
    /// event: the access is raw there.
    Raw(TableHandle),
    /// The table lacks the key and its handler is a Lua closure, which the
    /// access calls with this table.
    Closure(TableHandle, ClosureHandle),
    /// A handler that is a native or builtin function: the interpreter calls
    /// it, or a tier of the trace does.
    Native(Value),
    /// A handler that is none of those, a hop count past the interpreter's
    /// limit, or a stale handle: the interpreter answers.
    Other,
}

/// The walk of `event` for `key` from `start`.
pub(super) fn walk_handlers(
    runtime: &Runtime,
    start: TableHandle,
    key: &Value,
    event: &LuaString,
) -> Walk {
    let mut table = start;
    for _ in 0..crate::vm::MAX_INDEX_HOPS {
        // A string key is looked up without building a key.
        let held = match key {
            Value::String(text) => Ok(runtime.table_get_string_ref(table, text)),
            _ => runtime.table_get_raw_value(table, key),
        };
        match held {
            Ok(None) => {}
            Ok(Some(_)) => return Walk::Holds,
            Err(_) => return Walk::Other,
        }
        let metatable = match runtime.raw_metatable(&Value::Table(table)) {
            Ok(Some(metatable)) => metatable,
            Ok(None) => return Walk::Raw(table),
            Err(_) => return Walk::Other,
        };
        match runtime.table_get_string_ref(metatable, event) {
            None => return Walk::Raw(table),
            Some(Value::Table(next)) => table = *next,
            Some(Value::Closure(handler)) => return Walk::Closure(table, *handler),
            Some(handler @ (Value::NativeFunction(_) | Value::BuiltinFunction(_))) => {
                return Walk::Native(handler.clone());
            }
            Some(_) => return Walk::Other,
        }
    }
    Walk::Other
}

/// Whether `table` holds `key` itself, read raw. A string key is looked up
/// without building a key.
fn holds_key(runtime: &Runtime, table: TableHandle, key: &Value) -> Option<bool> {
    match key {
        Value::String(text) => Some(runtime.table_get_string_ref(table, text).is_some()),
        _ => runtime
            .table_get_raw_value(table, key)
            .ok()
            .map(|held| held.is_some()),
    }
}

/// The table and the Lua closure the walk of `event` for `key` from `start`
/// ends at, as [`walk_handlers`] finds them, for the checks that run in a
/// trace: a table without a metatable ends the walk without a look at the
/// key.
fn closure_of_walk(
    runtime: &mut Runtime,
    start: TableHandle,
    key: &Value,
    (number, event): (u8, &LuaString),
) -> Option<(TableHandle, ClosureHandle)> {
    let mut table = start;
    for _ in 0..crate::vm::MAX_INDEX_HOPS {
        let metatable = runtime.raw_metatable(&Value::Table(table)).ok().flatten()?;
        if holds_key(runtime, table, key)? {
            return None;
        }
        // The operator cache answers a closure or an absent handler without
        // a look at the metatable's keys.
        match runtime.operator_handler(metatable, number, event) {
            OperatorHandler::Closure(handler) => return Some((table, handler)),
            OperatorHandler::Absent => return None,
            OperatorHandler::Other => {}
        }
        match runtime.table_get_string_ref(metatable, event)? {
            Value::Table(next) => table = *next,
            Value::Closure(handler) => return Some((table, *handler)),
            _ => return None,
        }
    }
    None
}

/// The table a write of `key` from `start` stores in when no function runs:
/// the walk of `__newindex` through tables ends at one that has no metatable,
/// holds the key or has no handler. `None` for a walk that ends at anything
/// else.
fn store_target(runtime: &Runtime, start: TableHandle, key: &Value) -> Option<TableHandle> {
    let mut table = start;
    for _ in 0..crate::vm::MAX_INDEX_HOPS {
        let Some(metatable) = runtime.raw_metatable(&Value::Table(table)).ok()? else {
            return Some(table);
        };
        if holds_key(runtime, table, key)? {
            return Some(table);
        }
        match runtime.table_get_string_ref(metatable, newindex_metamethod_key()) {
            None => return Some(table),
            Some(Value::Table(next)) => table = *next,
            Some(_) => return None,
        }
    }
    None
}

/// The key a slot stands for, when it is one of the types a table access
/// takes in a trace: a string, an integer, a float, or a table or function
/// lane (a read under an object key, `ordinal_sturdy[shape]` in Mineclonia's
/// nodeprops.lua:705). `scratch` holds a number so the key can be borrowed.
#[allow(unsafe_code)]
#[inline]
fn slot_key<'a>(
    slot: TraceInputSlot,
    (bank, roots): (&'a [Value], &'a [Value]),
    scratch: &'a mut Value,
) -> Option<&'a Value> {
    match slot.tag() {
        TRACE_TAG_STRING | TRACE_TAG_BORROWED_STRING => {
            host::lane_value(slot, bank, roots).filter(|held| matches!(held, Value::String(_)))
        }
        TRACE_TAG_INTEGER => {
            *scratch = Value::Integer(slot.bits());
            Some(scratch)
        }
        TRACE_TAG_FLOAT => {
            *scratch = Value::Number(f64::from_bits(slot.bits().cast_unsigned()));
            Some(scratch)
        }
        TRACE_TAG_TABLE => {
            host::lane_value(slot, bank, roots).filter(|held| matches!(held, Value::Table(_)))
        }
        TRACE_TAG_CLOSURE => function_lane(bank, slot.bits()),
        _ => None,
    }
}

/// The function a closure lane of the bank holds: a Lua closure or a native
/// function, which a store clones like any other reference.
fn function_lane(bank: &[Value], lane: i64) -> Option<&Value> {
    match bank.get(usize::try_from(lane).ok()?)? {
        held @ (Value::Closure(_) | Value::NativeFunction(_) | Value::BuiltinFunction(_)) => {
            Some(held)
        }
        _ => None,
    }
}

/// The table a slot names, a lane of the run's bank.
fn slot_table(slot: TraceInputSlot, bank: &[Value]) -> Option<TableHandle> {
    match slot.tag() {
        TRACE_TAG_TABLE => match bank.get(usize::try_from(slot.bits()).ok()?)? {
            Value::Table(handle) => Some(*handle),
            _ => None,
        },
        _ => None,
    }
}

/// The check before a read or write that runs a function metamethod (see
/// [`HelperId::INDEX_HANDLER`] and [`HelperId::NEWINDEX_HANDLER`]): the walk
/// of `event` for the key ends at the pinned closure. Gives the lane of the
/// table the closure is called with. Changes nothing but the lanes of the
/// run's bank.
#[allow(unsafe_code)]
fn element_handler(
    context: usize,
    out: *mut i64,
    (table, key): ((i64, i64), (i64, i64)),
    identity: i64,
    (number, event): (u8, fn() -> &'static LuaString),
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let table_slot = TraceInputSlot::from_words(table.0, table.1);
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(start) = slot_table(table_slot, bank) else {
        return 1;
    };
    let mut scratch = Value::Nil;
    let Some(key) = slot_key(
        TraceInputSlot::from_words(key.0, key.1),
        (bank, roots),
        &mut scratch,
    ) else {
        return 1;
    };
    let holder = match closure_of_walk(runtime, start, key, (number, event())) {
        Some((holder, handler)) if handler.native_identity() == identity.cast_unsigned() => holder,
        _ => return 1,
    };
    // The receiver has its lane already; a table further down the chain
    // gets one.
    let lane = if holder == start {
        table.1
    } else {
        let Some(lane) = tables
            .lane(&Value::Table(holder))
            .and_then(|lane| i64::try_from(lane).ok())
        else {
            return 1;
        };
        lane
    };
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = lane };
    0
}

/// The check before a read that runs `__index` (see
/// [`HelperId::INDEX_HANDLER`]).
extern "C" fn helper_index_handler(
    context: usize,
    out: *mut i64,
    table_tag: i64,
    table_bits: i64,
    key_tag: i64,
    key_bits: i64,
    identity: i64,
) -> i64 {
    element_handler(
        context,
        out,
        ((table_tag, table_bits), (key_tag, key_bits)),
        identity,
        (event::INDEX, index_event_key),
    )
}

/// The check before a write that runs `__newindex` (see
/// [`HelperId::NEWINDEX_HANDLER`]).
extern "C" fn helper_newindex_handler(
    context: usize,
    out: *mut i64,
    table_tag: i64,
    table_bits: i64,
    key_tag: i64,
    key_bits: i64,
    identity: i64,
) -> i64 {
    element_handler(
        context,
        out,
        ((table_tag, table_bits), (key_tag, key_bits)),
        identity,
        (event::NEWINDEX, newindex_metamethod_key),
    )
}

/// The `__index` key, built once.
fn index_event_key() -> &'static LuaString {
    metamethod_key("__index").expect("__index is an operator metamethod")
}

/// The check before `#t` runs `__len` (see [`HelperId::LEN_HANDLER`]): the
/// value in the slot is a table whose metatable's raw `__len` is the Lua
/// closure of the identity the trace pinned. Changes nothing.
#[allow(unsafe_code)]
extern "C" fn helper_len_handler(
    context: usize,
    _out: *mut i64,
    tag: i64,
    bits: i64,
    handler_identity: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, _, _, _) = tables.library();
    let Some(table) = slot_table(TraceInputSlot::from_words(tag, bits), bank) else {
        return 1;
    };
    let Ok(Some(metatable)) = runtime.raw_metatable(&Value::Table(table)) else {
        return 1;
    };
    let key = metamethod_key("__len").expect("__len is an operator metamethod");
    match runtime.operator_handler(metatable, event::LEN, key) {
        OperatorHandler::Closure(handler) => {
            return i64::from(handler.native_identity() != handler_identity.cast_unsigned());
        }
        OperatorHandler::Absent => return 1,
        OperatorHandler::Other => {}
    }
    i64::from(!matches!(
        runtime.table_get_string_ref(metatable, key),
        Some(Value::Closure(handler))
            if handler.native_identity() == handler_identity.cast_unsigned()
    ))
}

/// The identity that asks for no handler: no closure has it (the generation
/// of a handle never fills its word).
pub(super) const OPERATOR_NO_HANDLER: i64 = -1;

/// The raw handler of the event `key` in the metatable of `value`, if the
/// value has one and it is not nil.
fn raw_handler<'a>(runtime: &'a Runtime, value: &Value, key: &LuaString) -> Option<&'a Value> {
    let Ok(Some(metatable)) = runtime.raw_metatable(value) else {
        return None;
    };
    runtime
        .table_get_string_ref(metatable, key)
        .filter(|handler| !matches!(handler, Value::Nil))
}

/// The handler an operator on `left` and `right` runs under `rule` (see
/// [`HelperId::OPERATOR_HANDLER`]), as `Vm::binary_metamethod`,
/// `first_comparison_metamethod`, `shared_metamethod` and
/// `ordered_metamethod` pick it.
pub(super) fn operator_handler<'a>(
    runtime: &'a Runtime,
    (left, right): (&Value, &Value),
    key: &LuaString,
    rule: i64,
) -> Option<&'a Value> {
    let first = raw_handler(runtime, left, key);
    match rule {
        // The left operand's handler, unless it is the string metatable's
        // arithmetic and the right operand is not a string: that handler
        // passes the operation to the right operand's own.
        0 => match first {
            Some(Value::NativeFunction(handler))
                if !matches!(right, Value::String(_))
                    && runtime
                        .direct_native_kind(*handler)
                        .is_some_and(DirectNativeKind::is_string_arithmetic) =>
            {
                raw_handler(runtime, right, key).or(first)
            }
            Some(_) => first,
            None => raw_handler(runtime, right, key),
        },
        1 => first.or_else(|| raw_handler(runtime, right, key)),
        2 | 3 => {
            if rule == 3 && left.type_name() != right.type_name() {
                return None;
            }
            let second = raw_handler(runtime, right, key)?;
            Runtime::raw_equal(first?, second).then_some(second)
        }
        _ => None,
    }
}

/// What the metatable of `value` holds under the operator `event`, through
/// the runtime's operator cache: a hit costs a compare and a load, and a miss
/// reads the metatable once for every later run.
fn cached_handler(
    runtime: &mut Runtime,
    value: &Value,
    event: u8,
    key: &LuaString,
) -> OperatorHandler {
    let Ok(Some(metatable)) = runtime.raw_metatable(value) else {
        return OperatorHandler::Absent;
    };
    runtime.operator_handler(metatable, event, key)
}

/// What [`cached_operator_handler`] makes of an operator's handlers.
enum CachedPick {
    /// The rule picks this Lua closure.
    Closure(ClosureHandle),
    /// The rule picks no handler.
    Nothing,
    /// A handler is another value, which only the full rule decides.
    Undecided,
}

/// [`operator_handler`] when every handler it looks at is absent or a Lua
/// closure, read through the operator cache.
fn cached_operator_handler(
    runtime: &mut Runtime,
    (left, right): (&Value, &Value),
    event: u8,
    key: &LuaString,
    rule: i64,
) -> CachedPick {
    let pick = |handler| match handler {
        OperatorHandler::Closure(handler) => CachedPick::Closure(handler),
        OperatorHandler::Absent => CachedPick::Nothing,
        OperatorHandler::Other => CachedPick::Undecided,
    };
    let first = cached_handler(runtime, left, event, key);
    match rule {
        0 | 1 => match first {
            OperatorHandler::Absent => pick(cached_handler(runtime, right, event, key)),
            first => pick(first),
        },
        2 | 3 => {
            if rule == 3 && left.type_name() != right.type_name() {
                return CachedPick::Nothing;
            }
            match (first, cached_handler(runtime, right, event, key)) {
                (_, OperatorHandler::Absent) | (OperatorHandler::Absent, _) => CachedPick::Nothing,
                (OperatorHandler::Closure(first), OperatorHandler::Closure(second)) => {
                    if first == second {
                        CachedPick::Closure(second)
                    } else {
                        CachedPick::Nothing
                    }
                }
                _ => CachedPick::Undecided,
            }
        }
        _ => CachedPick::Nothing,
    }
}

/// The check before an operator runs a Lua closure as its metamethod (see
/// [`HelperId::OPERATOR_HANDLER`]): the handler the rule picks for the two
/// operands is the closure of the identity the trace pinned, or none when the
/// identity is [`OPERATOR_NO_HANDLER`]. The pin keeps the closure alive, so no other closure can
/// take its identity. Changes nothing.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_operator_handler(
    context: usize,
    _out: *mut i64,
    left_tag: i64,
    left_bits: i64,
    right_tag: i64,
    right_bits: i64,
    selector: i64,
    handler_identity: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let slots = [
        TraceInputSlot::from_words(left_tag, left_bits),
        TraceInputSlot::from_words(right_tag, right_bits),
    ];
    let same = host::with_values(&slots, bank, roots, |values| {
        let [left, right] = values else {
            return false;
        };
        let Some((event, key)) = u8::try_from(selector & 0xff)
            .ok()
            .and_then(|event| Some((event, operator_event_key(event)?)))
        else {
            return false;
        };
        match cached_operator_handler(runtime, (left, right), event, key, selector >> 8) {
            CachedPick::Closure(handler) => {
                return handler.native_identity() == handler_identity.cast_unsigned();
            }
            CachedPick::Nothing => return handler_identity == OPERATOR_NO_HANDLER,
            CachedPick::Undecided => {}
        }
        let picked = operator_handler(runtime, (left, right), key, selector >> 8);
        match picked {
            None => handler_identity == OPERATOR_NO_HANDLER,
            Some(Value::Closure(handler)) => {
                handler.native_identity() == handler_identity.cast_unsigned()
            }
            Some(_) => false,
        }
    });
    i64::from(same != Some(true))
}

/// A store under a string key that runs no function (see
/// [`HelperId::TABLE_SET_STRING`]): the walk of `__newindex` ends in a table
/// that holds the key or lacks a handler (with `raw`, the table itself is
/// that table), and the runtime stores there as the interpreter's raw store
/// does, barriers and accounting included. It misses, having changed
/// nothing, for a key whose value a binding of the run fixes
/// ([`RunTables::pinned_keys`]).
#[allow(unsafe_code)]
extern "C" fn helper_table_set_string(
    context: usize,
    _out: *mut i64,
    table_tag: i64,
    table_bits: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
    raw: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let pinned = tables.pinned_keys();
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(start) = slot_table(TraceInputSlot::from_words(table_tag, table_bits), bank) else {
        return 1;
    };
    let key_slot = TraceInputSlot::from_words(key_tag, key_bits);
    if !matches!(key_slot.tag(), TRACE_TAG_STRING | TRACE_TAG_BORROWED_STRING) {
        return 1;
    }
    let mut scratch = Value::Nil;
    let Some(key @ Value::String(text)) = slot_key(key_slot, (bank, roots), &mut scratch) else {
        return 1;
    };
    if pinned.holds(text.as_bytes()) {
        return 1;
    }
    // The value is cloned out of the slot that holds it, so the store owns it.
    let value_slot = TraceInputSlot::from_words(value_tag, value_bits);
    let value = match value_slot.tag() {
        TRACE_TAG_NIL => Value::Nil,
        TRACE_TAG_BOOLEAN => Value::Boolean(value_bits != 0),
        TRACE_TAG_INTEGER => Value::Integer(value_bits),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(value_bits.cast_unsigned())),
        TRACE_TAG_STRING | TRACE_TAG_TABLE | TRACE_TAG_BORROWED_STRING => {
            match host::lane_value(value_slot, bank, roots) {
                Some(held @ (Value::String(_) | Value::Table(_))) => held.clone(),
                _ => return 1,
            }
        }
        TRACE_TAG_CLOSURE => match function_lane(bank, value_bits) {
            Some(held) => held.clone(),
            None => return 1,
        },
        _ => return 1,
    };
    // `rawset` stores in the table itself.
    let target = if raw != 0 {
        start
    } else {
        let Some(target) = store_target(runtime, start, key) else {
            return 1;
        };
        target
    };
    // Everything the store can refuse is checked first, so a miss changes
    // nothing.
    if pinned.holds_globals(target) || runtime.table_is_readonly(target).unwrap_or(true) {
        return 1;
    }
    let Value::String(text) = key else {
        return 1;
    };
    // A key that may name a metamethod changes what traces assumed about the
    // metatable at entry: the interpreter makes that store.
    if text.as_bytes().starts_with(b"__") {
        return 1;
    }
    let missed = if key_slot.tag() == TRACE_TAG_BORROWED_STRING {
        // A string borrowed from a table slot is cloned: the store may change
        // the table that holds it.
        let text = text.clone();
        store_string(runtime, target, &text, value, raw != 0)
    } else {
        store_string(runtime, target, text, value, raw != 0)
    };
    i64::from(missed)
}

/// A store under a number key into a table met by lane (see
/// [`HelperId::TABLE_SET_INDEX`]): the walk of `__newindex` ends in a table
/// that holds the key or lacks a handler, and the runtime stores there as the
/// interpreter's raw store does, barriers and accounting included. It misses,
/// having changed nothing, for a function `__newindex`, a read-only table, a
/// NaN key and a value the lanes cannot name.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_table_set_index(
    context: usize,
    _out: *mut i64,
    words: *mut TraceInputSlot,
    table_tag: i64,
    table_bits: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
    raw: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let pinned = tables.pinned_keys();
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(start) = slot_table(TraceInputSlot::from_words(table_tag, table_bits), bank) else {
        return 1;
    };
    let key_slot = TraceInputSlot::from_words(key_tag, key_bits);
    if !matches!(key_slot.tag(), TRACE_TAG_INTEGER | TRACE_TAG_FLOAT) {
        return 1;
    }
    let mut scratch = Value::Nil;
    let Some(key) = slot_key(key_slot, (bank, roots), &mut scratch) else {
        return 1;
    };
    if matches!(key, Value::Number(number) if number.is_nan()) {
        return 1;
    }
    // The value is cloned out of the slot that holds it, so the store owns it.
    let value_slot = TraceInputSlot::from_words(value_tag, value_bits);
    let value = match value_slot.tag() {
        TRACE_TAG_NIL => Value::Nil,
        TRACE_TAG_BOOLEAN => Value::Boolean(value_bits != 0),
        TRACE_TAG_INTEGER => Value::Integer(value_bits),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(value_bits.cast_unsigned())),
        TRACE_TAG_STRING | TRACE_TAG_TABLE | TRACE_TAG_BORROWED_STRING => {
            match host::lane_value(value_slot, bank, roots) {
                Some(held @ (Value::String(_) | Value::Table(_))) => held.clone(),
                _ => return 1,
            }
        }
        TRACE_TAG_CLOSURE => match function_lane(bank, value_bits) {
            Some(held) => held.clone(),
            None => return 1,
        },
        _ => return 1,
    };
    let target = if raw != 0 {
        start
    } else {
        let Some(target) = store_target(runtime, start, key) else {
            return 1;
        };
        target
    };
    // Everything the store can refuse is checked first, so a miss changes
    // nothing.
    if pinned.holds_globals(target) || runtime.table_is_readonly(target).unwrap_or(true) {
        return 1;
    }
    let key = key.clone();
    if runtime.table_set(target, &key, value).is_err() {
        return 1;
    }
    // The store may have grown the array part or moved the border of the
    // table it reached. A binding can name that table while the call reached
    // it by lane (the `t` a `__newindex` was handed), so the words of every
    // array and length binding of `target` are written again here, not left
    // for the next run.
    let view = tables.library().0.table_trace_array(target);
    if let Ok(view) = view {
        tables.rewrite_bindings(words, target, view);
    }
    0
}

/// Stores `value` under the string `text` of `target`, which takes it:
/// `true` for a miss. Without `raw` the table has no `__newindex` to run for
/// the key (the walk ended there), so the interpreter's own store applies.
fn store_string(
    runtime: &mut Runtime,
    target: TableHandle,
    text: &LuaString,
    value: Value,
    raw: bool,
) -> bool {
    if raw {
        runtime
            .table_set(target, &Value::String(text.clone()), value)
            .is_err()
    } else {
        !matches!(
            runtime.table_set_string_if_no_newindex(target, text, value),
            Ok(true)
        )
    }
}

/// The conversion of a string operand of arithmetic in the profiles before
/// Lua 5.3: a float, through the runtime's numeral spellings.
pub(crate) const STRING_NUMBER_FLOAT_MODE: i64 = 0;
/// The conversion in Lua 5.3: a float, converted through the integer numeral
/// if the string is one, as the interpreter does.
pub(crate) const STRING_NUMBER_LUA53_MODE: i64 = 1;
/// The first conversion of Lua 5.4 and later, where the string metatable's
/// arithmetic metamethod converts: an integer or a float as the numeral
/// reads. The mode adds the index of the operator in [`STRING_ARITHMETIC`].
pub(crate) const STRING_NUMBER_METAMETHOD_MODE: i64 = 2;

/// The metamethods of the string metatable that convert strings (Lua 5.4 and
/// later), in the order of the modes after [`STRING_NUMBER_METAMETHOD_MODE`].
pub(crate) const STRING_ARITHMETIC: [(&str, DirectNativeKind); 8] = [
    ("__add", DirectNativeKind::StringAdd),
    ("__sub", DirectNativeKind::StringSub),
    ("__mul", DirectNativeKind::StringMul),
    ("__div", DirectNativeKind::StringDiv),
    ("__mod", DirectNativeKind::StringMod),
    ("__pow", DirectNativeKind::StringPow),
    ("__idiv", DirectNativeKind::StringFloorDiv),
    ("__unm", DirectNativeKind::StringUnm),
];

thread_local! {
    /// The keys of [`STRING_ARITHMETIC`], made once.
    static STRING_ARITHMETIC_KEYS: [crate::string::LuaString; 8] =
        STRING_ARITHMETIC.map(|(name, _)| crate::string::LuaString::from(name));
}

/// The number the interpreter's arithmetic makes of the string `text` under
/// the conversion `mode`, or `None` where the interpreter does something else
/// (an error, a metamethod of another function, a number type with a
/// metatable). The recorder runs it on the live value, so a trace and the
/// helper that checks it agree with each other by construction.
pub(crate) fn string_number(runtime: &Runtime, mode: i64, text: &Value) -> Option<Value> {
    if !matches!(text, Value::String(_)) {
        return None;
    }
    match mode {
        STRING_NUMBER_FLOAT_MODE => {
            Runtime::coerce_number_in(text, "perform arithmetic", runtime.nonfinite_numerals())
                .ok()
                .map(Value::Number)
        }
        STRING_NUMBER_LUA53_MODE => {
            let numeric = Runtime::coerce_numeric_value(text, "perform arithmetic").ok()?;
            Runtime::coerce_number(&numeric, "perform arithmetic")
                .ok()
                .map(Value::Number)
        }
        mode => {
            let at = usize::try_from(mode - STRING_NUMBER_METAMETHOD_MODE).ok()?;
            let (_, kind) = *STRING_ARITHMETIC.get(at)?;
            let metatable = runtime.raw_metatable(text).ok()??;
            let handler = STRING_ARITHMETIC_KEYS
                .with(|keys| runtime.table_get_string_ref(metatable, &keys[at]).cloned());
            // The other operand is a number or a string. A number with a
            // metatable of its own might take the operation first.
            if !matches!(handler, Some(Value::NativeFunction(handler))
                if runtime.direct_native(kind) == Some(handler))
                || !matches!(runtime.raw_metatable(&Value::Integer(0)), Ok(None))
            {
                return None;
            }
            Runtime::coerce_numeric_value(text, "perform arithmetic").ok()
        }
    }
}

/// A string operand of arithmetic as a number (see
/// [`HelperId::STRING_NUMBER_INTEGER`]): `INTEGER` tells the kind of number
/// the recorded conversion gave. Changes nothing.
#[allow(unsafe_code)]
extern "C" fn helper_string_number<const INTEGER: bool>(
    context: usize,
    out: *mut i64,
    mode: i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let slot = TraceInputSlot::from_words(tag, bits);
    let number =
        host::lane_value(slot, bank, roots).and_then(|held| string_number(runtime, mode, held));
    let word = match number {
        Some(Value::Integer(value)) if INTEGER => value,
        Some(Value::Number(value)) if !INTEGER => value.to_bits().cast_signed(),
        _ => return 1,
    };
    // SAFETY: `out` has room for a helper's results.
    unsafe { *out = word };
    0
}

/// Which `math.random` the profile has, which tells the arguments it takes.
#[derive(Clone, Copy, PartialEq)]
enum Draw {
    /// Lua 5.1 and 5.2: bounds are floats in the range of `i32`.
    Float,
    /// `LuaJIT`: any numbers.
    LuaJit,
    /// Lua 5.3: integer bounds whose difference fits.
    Lua53,
    /// Lua 5.4 and 5.5: integer bounds, and `random (0)`.
    Lua54,
}

/// Whether the interpreter's `math.random` takes `args` without raising an
/// error. The generator moves before some of its checks, so the helper asks
/// first and the call it makes cannot fail.
fn random_takes(draw: Draw, args: &[TraceInputSlot]) -> bool {
    let integer = |slot: &TraceInputSlot| (slot.tag() == TRACE_TAG_INTEGER).then(|| slot.bits());
    let small = |slot: &TraceInputSlot| {
        let number = match slot.tag() {
            TRACE_TAG_FLOAT => f64::from_bits(slot.bits().cast_unsigned()),
            _ => return None,
        };
        #[allow(clippy::float_cmp, clippy::cast_possible_truncation)]
        (number.is_finite()
            && number.fract() == 0.0
            && number >= f64::from(i32::MIN)
            && number <= f64::from(i32::MAX))
        .then_some(number as i64)
    };
    match (draw, args) {
        (_, []) => true,
        (Draw::LuaJit, [_] | [_, _]) => args
            .iter()
            .all(|slot| matches!(slot.tag(), TRACE_TAG_FLOAT | TRACE_TAG_INTEGER)),
        (Draw::Float, [upper]) => small(upper).is_some_and(|upper| upper >= 1),
        (Draw::Float, [lower, upper]) => small(lower)
            .zip(small(upper))
            .is_some_and(|(lower, upper)| lower <= upper),
        (Draw::Lua53, [upper]) => integer(upper).is_some_and(|upper| upper >= 1),
        (Draw::Lua54, [upper]) => integer(upper).is_some_and(|upper| upper >= 0),
        (Draw::Lua53 | Draw::Lua54, [lower, upper]) => integer(lower)
            .zip(integer(upper))
            .is_some_and(|(lower, upper)| {
                lower <= upper && (draw == Draw::Lua54 || upper.checked_sub(lower).is_some())
            }),
        _ => false,
    }
}

/// `math.random` (see [`HelperId::MATH_RANDOM_FLOAT`]). `integer` is whether
/// the helper stands for the integer draw.
#[allow(unsafe_code)]
fn math_random(context: usize, out: *mut i64, args: &[TraceInputSlot], integer: bool) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let draw = if runtime.random_is_luajit() {
        Draw::LuaJit
    } else if runtime.random_is_modern() {
        Draw::Lua54
    } else if runtime.tagged_integers() {
        Draw::Lua53
    } else {
        Draw::Float
    };
    // Lua 5.3 and later give an integer for a bound, a float for none.
    let gives_integer = matches!(draw, Draw::Lua53 | Draw::Lua54) && !args.is_empty();
    if gives_integer != integer || !random_takes(draw, args) {
        return 1;
    }
    let mut slot = TraceInputSlot::other();
    if !host::through_native(
        runtime,
        bank,
        roots,
        (state, lengths),
        DirectNativeKind::MathRandom,
        args,
        &mut slot,
    ) {
        // The checks above promised an answer, and the generator has moved:
        // the interpreter's replay is the only thing left.
        debug_assert!(false, "a checked math.random call missed");
        return 1;
    }
    let word = match (integer, slot.tag()) {
        (true, TRACE_TAG_INTEGER) | (false, TRACE_TAG_FLOAT) => slot.bits(),
        _ => {
            debug_assert!(false, "math.random gave another type");
            return 1;
        }
    };
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = word };
    0
}

extern "C" fn helper_math_random_float(
    context: usize,
    out: *mut i64,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
    ];
    let count = usize::try_from(count).unwrap_or(3).min(3);
    if count > 2 {
        return 1;
    }
    math_random(context, out, &all[..count], false)
}

extern "C" fn helper_math_random_integer(
    context: usize,
    out: *mut i64,
    count: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
) -> i64 {
    let all = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
    ];
    let count = usize::try_from(count).unwrap_or(3).min(3);
    if count > 2 {
        return 1;
    }
    math_random(context, out, &all[..count], true)
}

/// `math.modf` of a float, as the interpreter's call gives it: the integral
/// part (an integer where the profile has them and the part fits) and the
/// fraction. `integer` is whether the helper stands for an integer part.
#[allow(
    unsafe_code,
    clippy::float_cmp,
    reason = "a zero fraction is exactly zero"
)]
fn math_modf(context: usize, out: *mut i64, value: f64, integer: bool) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, _, _, _, _) = tables.library();
    // The C `modf`'s fraction before Lua 5.3 has the sign of the argument,
    // zero included (`math_modf` of the library).
    let signed = !runtime.tagged_integers();
    let zero = if signed { 0.0_f64.copysign(value) } else { 0.0 };
    let (whole, fraction) = if value.is_nan() {
        (Value::Number(value), value)
    } else if !value.is_finite() {
        (Value::Number(value), zero)
    } else {
        let whole = value.trunc();
        let fraction = value - whole;
        (
            runtime.integral_float_value(whole),
            if fraction == 0.0 { zero } else { fraction },
        )
    };
    let word = match (integer, whole) {
        (true, Value::Integer(whole)) => whole,
        (false, Value::Number(whole)) => whole.to_bits().cast_signed(),
        _ => return 1,
    };
    // SAFETY: `out` holds a word per result (see `invoke`).
    unsafe {
        *out = word;
        *out.add(1) = fraction.to_bits().cast_signed();
    }
    0
}

extern "C" fn helper_math_modf_integer(context: usize, out: *mut i64, value: f64) -> i64 {
    math_modf(context, out, value, true)
}

extern "C" fn helper_math_modf_float(context: usize, out: *mut i64, value: f64) -> i64 {
    math_modf(context, out, value, false)
}

/// What `getmetatable` gives for the value in a slot.
enum Visible {
    /// The value has no metatable: nil.
    Nothing,
    /// A metatable without a `__metatable` field: the table itself.
    Table(TableHandle),
    /// The interpreter must answer: the `__metatable` field is the answer, or
    /// the slot holds no value a helper reads.
    Interpreter,
}

/// The metatable `getmetatable` gives for the value in the slot.
fn visible_metatable(tables: &mut RunTables<'static>, slot: TraceInputSlot) -> Visible {
    let (runtime, bank, roots, _, _) = tables.library();
    let found = host::with_values(&[slot], bank, roots, |values| {
        runtime.raw_metatable(&values[0]).ok()
    })
    .flatten();
    let Some(found) = found else {
        return Visible::Interpreter;
    };
    let Some(metatable) = found else {
        return Visible::Nothing;
    };
    if runtime.metatable_lacks(metatable, Metamethod::Metatable) {
        Visible::Table(metatable)
    } else {
        Visible::Interpreter
    }
}

/// `getmetatable` of a value that has a metatable: its lane in the run's
/// bank, which keeps it alive.
#[allow(unsafe_code)]
extern "C" fn helper_getmetatable(context: usize, out: *mut i64, tag: i64, bits: i64) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let Visible::Table(metatable) =
        visible_metatable(tables, TraceInputSlot::from_words(tag, bits))
    else {
        return 1;
    };
    let Some(lane) = tables
        .lane(&Value::Table(metatable))
        .and_then(|lane| i64::try_from(lane).ok())
    else {
        return 1;
    };
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = lane };
    0
}

/// `getmetatable` of a value that has none: nil. Misses when there is one.
#[allow(unsafe_code)]
extern "C" fn helper_getmetatable_absent(
    context: usize,
    _out: *mut i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    i64::from(!matches!(
        visible_metatable(tables, TraceInputSlot::from_words(tag, bits)),
        Visible::Nothing
    ))
}

/// `setmetatable (table, metatable)` for a table whose metatable is not
/// protected and a new one without `__gc` or `__mode`, which the interpreter's
/// call registers a finalizer or a weak table for. The checks come first, so a
/// miss changes nothing; the store itself is the interpreter's
/// (`Runtime::set_table_metatable`, with its collector barrier).
#[allow(unsafe_code)]
extern "C" fn helper_setmetatable(
    context: usize,
    _out: *mut i64,
    table: i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, _, _, _) = tables.library();
    let Some(Value::Table(target)) = usize::try_from(table).ok().and_then(|lane| bank.get(lane))
    else {
        return 1;
    };
    let target = *target;
    let metatable = match tag {
        TRACE_TAG_NIL => None,
        TRACE_TAG_TABLE => match usize::try_from(bits).ok().and_then(|lane| bank.get(lane)) {
            Some(Value::Table(metatable)) => Some(*metatable),
            _ => return 1,
        },
        _ => return 1,
    };
    if runtime.table_is_readonly(target).unwrap_or(true) {
        return 1;
    }
    match runtime.raw_metatable(&Value::Table(target)) {
        Ok(Some(current)) if !runtime.metatable_lacks(current, Metamethod::Metatable) => return 1,
        Ok(_) => {}
        Err(_) => return 1,
    }
    if let Some(metatable) = metatable
        && (!runtime.metatable_lacks(metatable, Metamethod::Gc)
            || !runtime.metatable_lacks(metatable, Metamethod::Mode))
    {
        return 1;
    }
    i64::from(runtime.set_table_metatable(target, metatable).is_err())
}

/// `table[key] = value` for a string key (see [`HelperId::SET_STRING`]): the
/// interpreter's raw store, which adds, overwrites or deletes the key. It
/// misses, having changed nothing, for a key or table the lanes do not hold, a
/// value that is no scalar, a read-only table, an absent key of a table
/// whose metatable has `__newindex`, a key that may name a metamethod, and a
/// key whose value a binding of the run fixes ([`RunTables::pinned_keys`]).
#[allow(unsafe_code)]
extern "C" fn helper_set_string(
    context: usize,
    _out: *mut i64,
    table: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.set_string += 1;
    let pinned = tables.pinned_keys();
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(&Value::Table(handle)) = usize::try_from(table).ok().and_then(|lane| bank.get(lane))
    else {
        return 1;
    };
    let mut scratch = Value::Nil;
    let key_slot = TraceInputSlot::from_words(key_tag, key_bits);
    let (Some(Value::String(key)), Some(value)) = (
        host::lane_value(key_slot, bank, roots),
        stored_value(
            TraceInputSlot::from_words(value_tag, value_bits),
            bank,
            roots,
            &mut scratch,
        ),
    ) else {
        return 1;
    };
    // A read-only table is no check of its own: the store refuses it, and the
    // miss hands the store to the interpreter, which raises the error.
    if names_metamethod(key) || pinned.holds(key.as_bytes()) || pinned.holds_globals(handle) {
        return 1;
    }
    // A scalar over a present scalar of its type is an overwrite in place,
    // which a version of the trace recorded now makes natively: the run
    // counts them (see `TRACE_OVERWRITE_VERSION_CALLS`).
    if matches!(
        value,
        Value::Boolean(_) | Value::Integer(_) | Value::Number(_)
    ) && matches!(
        runtime.table_overwrite_scalar_string(handle, key, value),
        Ok(true)
    ) {
        tables.store_calls.set_string_overwrites += 1;
        return 0;
    }
    let stored = if key_slot.tag() == TRACE_TAG_BORROWED_STRING {
        // A string borrowed from a table slot is cloned: the store may change
        // the table that holds it.
        let key = key.clone();
        runtime.table_set_string_from(handle, &key, value)
    } else {
        runtime.table_set_string_from(handle, key, value)
    };
    i64::from(!matches!(stored, Ok(true)))
}

/// [`HelperId::TABLE_NEW`]: makes the table a virtual one stood for, as the
/// interpreter's constructor would (with the shape the keys name, when there
/// are any), and gives its lane. The run's bank roots it until the next
/// release. A miss changes nothing a guest can see: a lane or a table that
/// was made stays unnamed, for the next release or the collector.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_table_new(
    context: usize,
    out: *mut i64,
    count: i64,
    array_len: i64,
    tag0: i64,
    bits0: i64,
    tag1: i64,
    bits1: i64,
    tag2: i64,
    bits2: i64,
    tag3: i64,
    bits3: i64,
    tag4: i64,
    bits4: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.table_new += 1;
    let prefill = count & TABLE_NEW_PREFILL != 0;
    if count & (TABLE_NEW_PREFILL - 1) <= 1 {
        return table_new_plain(tables, out, prefill, array_len);
    }
    table_new_keyed(
        context, out, count, array_len, tag0, bits0, tag1, bits1, tag2, bits2, tag3, bits3, tag4,
        bits4,
    )
}

/// [`helper_table_new`] for a call that names keys. It takes the helper's
/// own arguments in the helper's order, so the helper leaves for it with a
/// jump and no word is copied.
#[allow(unsafe_code, clippy::too_many_arguments)]
#[inline(never)]
extern "C" fn table_new_keyed(
    context: usize,
    out: *mut i64,
    count: i64,
    array_len: i64,
    tag0: i64,
    bits0: i64,
    tag1: i64,
    bits1: i64,
    tag2: i64,
    bits2: i64,
    tag3: i64,
    bits3: i64,
    tag4: i64,
    bits4: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let prefill = count & TABLE_NEW_PREFILL != 0;
    let count = count & (TABLE_NEW_PREFILL - 1);
    table_new_shaped(
        tables,
        out,
        (count, prefill, array_len),
        [
            tag0, bits0, tag1, bits1, tag2, bits2, tag3, bits3, tag4, bits4,
        ],
    )
}

/// [`helper_table_new`] for a call that names no key: a list constructor
/// (`{a, b}`) or an empty table. With `prefill` and a length the emitter
/// fills itself, the table is born with its array part made of nil slots.
/// The checks are those of [`table_new_shaped`], in its order: a spare slot
/// in the arena, no collection owed, a lane, then the table. Out of line,
/// so this call has a small frame of its own and the helper only jumps
/// here.
#[allow(unsafe_code)]
#[inline(never)]
fn table_new_plain(
    tables: &mut RunTables<'static>,
    out: *mut i64,
    prefill: bool,
    array_len: i64,
) -> i64 {
    let (runtime, _, _, _, _) = tables.library();
    if !runtime.table_slot_is_spare() || runtime.gc_needs_collection() {
        return 1;
    }
    let Some(lane) = tables.reserve_made_lane() else {
        return 1;
    };
    let (runtime, _, _, _, _) = tables.library();
    let length = usize::try_from(array_len).unwrap_or(0);
    let made = if prefill && (1..=MAX_PREFILLED_ARRAY).contains(&length) {
        runtime
            .create_table_prefilled_inline(length)
            .map(|(handle, slots)| (handle, Some(slots)))
    } else {
        runtime
            .create_table_sized(length.min(64))
            .ok()
            .map(|handle| (handle, None))
    };
    let Some((handle, slots)) = made else {
        tables.return_made_lane(lane);
        return 1;
    };
    let Some(lane) = tables
        .fill_made_lane(lane, handle)
        .and_then(|lane| i64::try_from(lane).ok())
    else {
        return 1;
    };
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`), which
    // has room for more than the lane.
    unsafe {
        *out = lane;
        if let Some(slots) = slots {
            *out.add(1) = slots as i64;
        }
    }
    0
}

/// [`helper_table_new`] for a call that names keys: the table is made with
/// their shape. In line in [`table_new_keyed`], so the words are read where
/// they arrive.
///
/// The checks come first, in the order of [`table_new_plain`]: a spare slot
/// in the arena (a table made with the arena full would move the tables
/// native code holds the addresses of), no collection owed (a table made
/// while the collector is owed a step would pile debt up for the whole
/// slice: the miss leaves before the constructor, where the interpreter
/// takes the safepoint), and a lane (a table with none would be of no use).
/// Nothing is made when one fails.
#[allow(unsafe_code, clippy::inline_always)]
#[inline(always)]
fn table_new_shaped(
    tables: &mut RunTables<'static>,
    out: *mut i64,
    (count, prefill, array_len): (i64, bool, i64),
    words: [i64; 10],
) -> i64 {
    let given = usize::try_from(count).unwrap_or(0).saturating_sub(1).min(5);
    // The shape of a call is built once per trace: the words of its keys
    // name it. A call's words are compared one by one against each shape's
    // (the words past the keys are zero), and a trace has a few shapes, so
    // the search is a handful of compares.
    let shape = if given == 0 {
        None
    } else {
        let named = &words[..given * 2];
        let (_, _, _, host_state, _) = tables.library();
        let known = host_state.shapes.iter().position(|(known, _)| {
            let mut same = true;
            for (index, word) in known.iter().enumerate() {
                same &= *word == named.get(index).copied().unwrap_or(0);
            }
            same
        });
        match known.or_else(|| table_new_shape(tables, given, &words)) {
            Some(position) => Some(position),
            None => return 1,
        }
    };
    let (runtime, _, _, _, _) = tables.library();
    if !runtime.table_slot_is_spare() || runtime.gc_needs_collection() {
        return 1;
    }
    let Some(lane) = tables.reserve_made_lane() else {
        return 1;
    };
    let (runtime, _, _, host_state, _) = tables.library();
    // Native code fills the slots a shape names in place (see
    // `HelperId::TABLE_INIT`): they stay where they are while the run only
    // fills them.
    let made = match shape {
        Some(position) => runtime
            .create_table_with_shape_slots(&host_state.shapes[position].1)
            .map(|(handle, slots)| (handle, Some(slots))),
        None => runtime
            .create_table_sized(usize::try_from(array_len).unwrap_or(0).min(64))
            .ok()
            .map(|handle| {
                let slots = if prefill {
                    usize::try_from(array_len)
                        .ok()
                        .filter(|count| (1..=MAX_PREFILLED_ARRAY).contains(count))
                        .and_then(|count| runtime.table_prefill_array(handle, count))
                } else {
                    None
                };
                (handle, slots)
            }),
    };
    let Some((handle, slots)) = made else {
        tables.return_made_lane(lane);
        return 1;
    };
    let Some(lane) = tables
        .fill_made_lane(lane, handle)
        .and_then(|lane| i64::try_from(lane).ok())
    else {
        return 1;
    };
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`), which
    // has room for more than the lane.
    unsafe {
        *out = lane;
        if let Some(slots) = slots {
            *out.add(1) = slots as i64;
        }
    }
    0
}

/// Builds the shape of a [`helper_table_new`] call that names `given` keys
/// in `words`, the first time the trace makes it, and gives its place among
/// the trace's shapes. `None` when a word names no string or the keys make
/// no shape. The shape is pinned while the process still pins shapes, so the
/// tables made with it count themselves neither in nor out of it.
#[cold]
#[inline(never)]
fn table_new_shape(
    tables: &mut RunTables<'static>,
    given: usize,
    words: &[i64; 10],
) -> Option<usize> {
    let (_, bank, roots, host_state, _) = tables.library();
    let mut key = [0_i64; 10];
    key[..given * 2].copy_from_slice(&words[..given * 2]);
    let mut names = Vec::with_capacity(given);
    for position in 0..given {
        let slot = TraceInputSlot::from_words(words[position * 2], words[position * 2 + 1]);
        let Some(Value::String(name)) = host::lane_value(slot, bank, roots) else {
            return None;
        };
        names.push(name.clone());
    }
    let shape = crate::runtime::TableShape::new(names)?;
    shape.pin();
    host_state.shapes.push((key, shape));
    Some(host_state.shapes.len() - 1)
}

/// The closure a lane of the run's bank holds, if it holds a Lua closure.
fn lane_closure(bank: &[Value], lane: i64) -> Option<ClosureHandle> {
    match bank.get(usize::try_from(lane).ok()?)? {
        Value::Closure(handle) => Some(*handle),
        _ => None,
    }
}

/// [`HelperId::CLOSURE_SAME_PROTO`]: whether the closures of two lanes share
/// a chunk, which is what makes them one prototype. A lane that holds no Lua
/// closure (a native function) is no match.
#[allow(unsafe_code)]
extern "C" fn helper_closure_same_proto(
    context: usize,
    out: *mut i64,
    live: i64,
    pinned: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.closure += 1;
    let (runtime, bank, _, _, _) = tables.library();
    let same = match (lane_closure(bank, live), lane_closure(bank, pinned)) {
        (Some(live), Some(pinned)) => {
            match (
                runtime.closure_call_parts(live),
                runtime.closure_call_parts(pinned),
            ) {
                (Some(live), Some(pinned)) => std::sync::Arc::ptr_eq(live.chunk, pinned.chunk),
                _ => false,
            }
        }
        _ => false,
    };
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = i64::from(same) };
    0
}

/// [`HelperId::CLOSURE_SAME_ENV`]: whether the closures of two lanes have one
/// environment table.
#[allow(unsafe_code)]
extern "C" fn helper_closure_same_env(
    context: usize,
    out: *mut i64,
    live: i64,
    pinned: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.closure += 1;
    let (runtime, bank, _, _, _) = tables.library();
    let environment = |lane: i64| {
        lane_closure(bank, lane)
            .and_then(|handle| runtime.closure_call_parts(handle))
            .map(|parts| parts.environment)
    };
    let same = matches!((environment(live), environment(pinned)), (Some(a), Some(b)) if a == b);
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = i64::from(same) };
    0
}

/// [`HelperId::TABLE_SAME`]: whether the tables of two lanes are one table. A
/// function trace's run gives a table a lane at every read, so this tells
/// them apart by the handle the bank holds, as `rawequal` does.
#[allow(unsafe_code)]
extern "C" fn helper_table_same(context: usize, out: *mut i64, left: i64, right: i64) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (_, bank, _, _, _) = tables.library();
    let table = |lane: i64| match bank.get(usize::try_from(lane).ok()?)? {
        Value::Table(handle) => Some(*handle),
        _ => None,
    };
    let same = left == right || matches!((table(left), table(right)), (Some(a), Some(b)) if a == b);
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = i64::from(same) };
    0
}

/// What a cell read wants to find.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CellKind {
    Integer,
    Float,
    Boolean,
    Table,
    /// A Lua closure or a native, as its lane.
    Callable,
    /// A string, as a lane of the bank that holds it.
    String,
}

/// What tells a value a cell read of `kind` answers for from every other
/// value of that type the run holds: the number's bits, or the identity of
/// the object. `None` for a value of another type, and for a native, whose
/// lane is the first the bank has for it, which the memo does not track.
#[inline(always)]
fn cell_identity(kind: CellKind, value: &Value) -> Option<u64> {
    Some(match (kind, value) {
        (CellKind::Integer, Value::Integer(value)) => value.cast_unsigned(),
        (CellKind::Float, Value::Number(value)) => value.to_bits(),
        (CellKind::Boolean, Value::Boolean(value)) => u64::from(*value),
        (CellKind::Table, Value::Table(table)) => table.native_identity(),
        (CellKind::Callable, Value::Closure(closure)) => closure.native_identity(),
        (CellKind::String, Value::String(text)) => text.identity() as u64,
        _ => return None,
    })
}

/// How many cell reads, and how many lanes of table reads, a run remembers
/// ([`ReadMemo`]).
const MEMO_PLACES: usize = 64;

/// The place of a word among [`MEMO_PLACES`], by Fibonacci hashing: the
/// lanes of a loop's closures are scattered over the bank, and the low bits
/// of a lane or an identity alone collide.
#[inline(always)]
fn memo_place(key: u64) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - MEMO_PLACES.trailing_zeros())) as usize
}

/// One remembered cell read: upvalue `index` of the closure in lane `lane`,
/// read for `kind`, held the value with `identity` and answered `word`.
#[derive(Clone, Copy)]
struct CellRead {
    /// The run that made the read ([`LaneIndex`](super::native::LaneIndex)'s
    /// stamp); an entry of another run is free.
    stamp: u32,
    lane: u32,
    index: u32,
    kind: CellKind,
    /// The identity of the closure in `lane`, and its cell.
    closure: u64,
    cell: Option<CellHandle>,
    /// [`Runtime::cell_writes`] when the read was made.
    writes: u64,
    identity: u64,
    word: i64,
}

/// An object a read found, which answers with its lane.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    Table(TableHandle),
    Closure(ClosureHandle),
    Userdata(UserdataHandle),
}

impl Held {
    /// The object `value` is, for the kinds a read gives a lane.
    #[inline(always)]
    fn of(value: &Value) -> Option<Self> {
        Some(match value {
            Value::Table(table) => Self::Table(*table),
            Value::Closure(closure) => Self::Closure(*closure),
            Value::Userdata(userdata) => Self::Userdata(*userdata),
            _ => return None,
        })
    }

    #[inline(always)]
    fn value(self) -> Value {
        match self {
            Self::Table(table) => Value::Table(table),
            Self::Closure(closure) => Value::Closure(closure),
            Self::Userdata(userdata) => Value::Userdata(userdata),
        }
    }

    /// A word that tells objects apart, to place them in the memo.
    #[inline(always)]
    fn key(self) -> u64 {
        match self {
            Self::Table(table) => table.native_identity(),
            Self::Closure(closure) => closure.native_identity() ^ (1 << 63),
            Self::Userdata(userdata) => u64::from(userdata.index()) ^ (1 << 62),
        }
    }
}

/// One remembered lane: the run gave `held` lane `lane`.
#[derive(Clone, Copy)]
struct HeldLane {
    /// The run that asked, as for [`CellRead`].
    stamp: u32,
    lane: u32,
    held: Option<Held>,
}

/// What a run remembers of its reads through the helpers. Both memos are
/// direct-mapped and stamped with the run, so a run starts with them empty
/// without clearing them.
///
/// `cells` holds the cell reads ([`HelperId::CELL_GET_INTEGER`] and its
/// siblings) by closure lane and upvalue index. A loop that reads the same
/// upvalues of the same few closures on every trip answers from here while
/// nothing it relies on has moved:
///
/// - the lane still holds the closure, which roots its cells for the run;
/// - no closure's cell was replaced: [`Runtime::cell_writes`] stands where it
///   stood (`set_closure_cell` and `reopen_cell` move it);
/// - the cell still holds the same value. The cell is read again, open or
///   closed, so a store the count does not see (native code's own store to a
///   callee cell, a register write through an open cell) still shows;
/// - the answer's lane, for a table, a closure or a string, still holds that
///   value: a release may have emptied it and handed it to another.
///
/// `lanes` holds the lane each table, closure or userdata a table read found
/// was given ([`HelperId::TABLE_GET_TABLE`] and its siblings), which answers
/// while the lane still holds the object. The run gives one object one lane
/// (the bank roots it, and a lane the run made a table in is kept once a
/// read finds it there), so the lane it gave is the lane it would give again.
///
/// A function trace's run gives a table a fresh lane at every read, so its
/// table reads are remembered in neither.
pub(super) struct ReadMemo {
    cells: [CellRead; MEMO_PLACES],
    lanes: [HeldLane; MEMO_PLACES],
}

impl Default for ReadMemo {
    fn default() -> Self {
        Self {
            cells: [CellRead {
                stamp: 0,
                lane: u32::MAX,
                index: u32::MAX,
                kind: CellKind::Integer,
                closure: 0,
                cell: None,
                writes: 0,
                identity: 0,
                word: 0,
            }; MEMO_PLACES],
            lanes: [HeldLane {
                stamp: 0,
                lane: u32::MAX,
                held: None,
            }; MEMO_PLACES],
        }
    }
}

impl ReadMemo {
    #[inline(always)]
    fn place(lane: i64, index: i64) -> usize {
        memo_place((lane.cast_unsigned() << 8) ^ index.cast_unsigned())
    }
}

/// The lane of the object `held`, which a table read found: the one the run
/// gave it before, while that lane still holds it, or else the one
/// [`TraceTables::lane`] gives, which the run remembers.
#[inline(always)]
fn held_lane(tables: &mut RunTables<'static>, held: Held) -> Option<i64> {
    let remember = !(matches!(held, Held::Table(_)) && tables.admits_fresh());
    if remember {
        let (_, bank, memo, stamp) = tables.read_memo();
        let known = memo.lanes[memo_place(held.key())];
        if known.stamp == stamp
            && known.held == Some(held)
            && bank
                .get(known.lane as usize)
                .is_some_and(|value| Held::of(value) == Some(held))
        {
            return Some(i64::from(known.lane));
        }
    }
    let lane = tables.lane(&held.value())?;
    let word = i64::try_from(lane).ok()?;
    if remember && let Ok(lane) = u32::try_from(lane) {
        let (_, _, memo, stamp) = tables.read_memo();
        memo.lanes[memo_place(held.key())] = HeldLane {
            stamp,
            lane,
            held: Some(held),
        };
    }
    Some(word)
}

/// The answer of a remembered read of upvalue `index` of the closure in lane
/// `closure` (see [`ReadMemo`]), or `None` when the run must read again.
#[inline(always)]
#[allow(unsafe_code)]
fn cell_memo_hit(
    tables: &mut RunTables<'static>,
    closure: i64,
    index: i64,
    kind: CellKind,
) -> Option<i64> {
    let (runtime, bank, memo, stamp) = tables.read_memo();
    let read = &memo.cells[ReadMemo::place(closure, index)];
    if read.stamp != stamp
        || i64::from(read.lane) != closure
        || i64::from(read.index) != index
        || read.kind != kind
        || read.writes != runtime.cell_writes()
    {
        return None;
    }
    match bank.get(read.lane as usize) {
        Some(Value::Closure(held)) if held.native_identity() == read.closure => {}
        _ => return None,
    }
    let cell = read.cell?;
    // SAFETY: the closure in the lane, which the bank roots, held this cell
    // when the read was made, and no closure's cell was replaced since (the
    // count of cell writes stands still), so the cell is alive.
    let value = unsafe { runtime.cell_ref_live(cell) };
    if cell_identity(kind, value) != Some(read.identity) {
        return None;
    }
    if matches!(
        kind,
        CellKind::Table | CellKind::Callable | CellKind::String
    ) {
        let held = bank.get(usize::try_from(read.word).ok()?)?;
        if cell_identity(kind, held) != Some(read.identity) {
            return None;
        }
    }
    Some(read.word)
}

/// The cell upvalue `index` of the closure in lane `closure` has, and the
/// value it holds now, when the closure and its cell exist. An open cell
/// lends its stack slot, so this sees what the interpreter's own
/// `GetUpvalue` would.
fn cell_value(
    tables: &mut RunTables<'static>,
    closure: i64,
    index: i64,
) -> Option<(ClosureHandle, CellHandle, Value)> {
    let (runtime, bank, _, _, _) = tables.library();
    let handle = lane_closure(bank, closure)?;
    let cell = runtime
        .closure_cell(handle, usize::try_from(index).ok()?)
        .ok()
        .flatten()?;
    Some((handle, cell, runtime.cell_ref(cell).ok()?.clone()))
}

/// [`HelperId::CELL_GET_INTEGER`] and its siblings: reads a cell through the
/// live closure and misses, having changed nothing, for another type.
///
/// Most reads are answered by the run's memo. That part is inlined into each
/// helper, and the read itself ([`cell_get_read`]) is a call of its own, so a
/// memo hit does not save and restore the registers the read needs.
#[allow(unsafe_code)]
#[inline(always)]
fn cell_get(context: usize, out: *mut i64, closure: i64, index: i64, kind: CellKind) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.closure += 1;
    if let Some(word) = cell_memo_hit(tables, closure, index, kind) {
        // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
        unsafe { *out = word };
        return 0;
    }
    cell_get_read(tables, out, closure, index, kind)
}

/// The read of [`cell_get`] that the memo did not answer, which it then
/// remembers.
#[allow(unsafe_code)]
#[inline(never)]
fn cell_get_read(
    tables: &mut RunTables<'static>,
    out: *mut i64,
    closure: i64,
    index: i64,
    kind: CellKind,
) -> i64 {
    let (runtime, _, _, _, _) = tables.library();
    let writes = runtime.cell_writes();
    let Some((handle, cell, value)) = cell_value(tables, closure, index) else {
        return 1;
    };
    let identity = cell_identity(kind, &value);
    let word = match (kind, value) {
        (CellKind::Integer, Value::Integer(value)) => value,
        (CellKind::Float, Value::Number(value)) => value.to_bits().cast_signed(),
        (CellKind::Boolean, Value::Boolean(value)) => i64::from(value),
        (CellKind::Table, Value::Table(table)) => {
            let Some(lane) = tables
                .lane(&Value::Table(table))
                .and_then(|lane| i64::try_from(lane).ok())
            else {
                return 1;
            };
            lane
        }
        (CellKind::Callable, Value::Closure(closure)) => {
            let Some(lane) = tables
                .lane(&Value::Closure(closure))
                .and_then(|lane| i64::try_from(lane).ok())
            else {
                return 1;
            };
            lane
        }
        // A native has the lane the trace pinned it in: the call guard
        // compares lanes, and one the trace did not pin is no callee it
        // recorded.
        (CellKind::Callable, held @ Value::NativeFunction(_)) => {
            let (_, bank, _, _, _) = tables.library();
            let Some(lane) = bank
                .iter()
                .position(|known| *known == held)
                .and_then(|lane| i64::try_from(lane).ok())
            else {
                return 1;
            };
            lane
        }
        // A string goes into a lane of the bank, as a table read's does
        // (`table_read`): the bank roots it for the rest of the run.
        (CellKind::String, held @ Value::String(_)) => {
            let (_, bank, _, state, lengths) = tables.library();
            let mut slot = TraceInputSlot::other();
            if !host::publish(bank, state, lengths, held, &mut slot)
                || slot.tag() != TRACE_TAG_STRING
            {
                return 1;
            }
            slot.bits()
        }
        _ => return 1,
    };
    let fresh = kind == CellKind::Table && tables.admits_fresh();
    if let (Some(identity), Ok(lane), Ok(upvalue), false) = (
        identity,
        u32::try_from(closure),
        u32::try_from(index),
        fresh,
    ) {
        let (_, _, memo, stamp) = tables.read_memo();
        memo.cells[ReadMemo::place(closure, index)] = CellRead {
            stamp,
            lane,
            index: upvalue,
            kind,
            closure: handle.native_identity(),
            cell: Some(cell),
            writes,
            identity,
            word,
        };
    }
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = word };
    0
}

extern "C" fn helper_cell_get_string(
    context: usize,
    out: *mut i64,
    closure: i64,
    index: i64,
) -> i64 {
    cell_get(context, out, closure, index, CellKind::String)
}

extern "C" fn helper_cell_get_callable(
    context: usize,
    out: *mut i64,
    closure: i64,
    index: i64,
) -> i64 {
    cell_get(context, out, closure, index, CellKind::Callable)
}

extern "C" fn helper_cell_get_integer(
    context: usize,
    out: *mut i64,
    closure: i64,
    index: i64,
) -> i64 {
    cell_get(context, out, closure, index, CellKind::Integer)
}

extern "C" fn helper_cell_get_float(
    context: usize,
    out: *mut i64,
    closure: i64,
    index: i64,
) -> i64 {
    cell_get(context, out, closure, index, CellKind::Float)
}

extern "C" fn helper_cell_get_boolean(
    context: usize,
    out: *mut i64,
    closure: i64,
    index: i64,
) -> i64 {
    cell_get(context, out, closure, index, CellKind::Boolean)
}

extern "C" fn helper_cell_get_table(
    context: usize,
    out: *mut i64,
    closure: i64,
    index: i64,
) -> i64 {
    cell_get(context, out, closure, index, CellKind::Table)
}

/// What a [`HelperId::TABLE_GET_INTEGER`] family helper hands back.
#[derive(Clone, Copy)]
enum ReadKind {
    Integer,
    Float,
    Boolean,
    Table,
    Closure,
    Userdata,
    String,
}

/// The value of `key` as `rawget` finds it from `start`, following the chain
/// of tables `__index` names when `walk` and the key is absent. `None` for a
/// key that is absent or nil in the table the read ends at, a handler that is
/// not a table, or a chain past the interpreter's hop limit. The value is
/// borrowed from the table that holds it: a read answers with a word or a
/// lane, and only a string needs a clone of its own.
#[inline]
pub(super) fn table_read_value<'r>(
    runtime: &'r Runtime,
    start: TableHandle,
    key: &Value,
    walk: bool,
) -> Option<&'r Value> {
    let mut table = start;
    for _ in 0..crate::vm::MAX_INDEX_HOPS {
        let held = match key {
            Value::String(text) => runtime.table_get_string_ref(table, text),
            _ => runtime.table_get_raw_hit(table, key),
        };
        match held {
            Some(Value::Nil) | None => {}
            Some(found) => return Some(found),
        }
        if !walk {
            return None;
        }
        // Absent (or nil): the interpreter asks the metatable's `__index`.
        let metatable = runtime.raw_metatable(&Value::Table(table)).ok().flatten()?;
        match runtime.table_get_string_ref(metatable, index_event_key())? {
            Value::Table(next) => table = *next,
            _ => return None,
        }
    }
    None
}

/// The family of [`HelperId::TABLE_GET_INTEGER`]: reads a table met by lane
/// and misses, having changed nothing, for anything the interpreter must
/// see.
#[allow(unsafe_code)]
fn table_read(
    context: usize,
    out: *mut i64,
    (table, key): ((i64, i64), (i64, i64)),
    walk: i64,
    kind: ReadKind,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.table_lookup += 1;
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(start) = slot_table(TraceInputSlot::from_words(table.0, table.1), bank) else {
        return 1;
    };
    let mut scratch = Value::Nil;
    let Some(key) = slot_key(
        TraceInputSlot::from_words(key.0, key.1),
        (bank, roots),
        &mut scratch,
    ) else {
        return 1;
    };
    let hit =
        table_read_value(runtime, start, key, walk != 0).and_then(|found| read_hit(kind, found));
    let word = match hit {
        Some(ReadHit::Word(word)) => word,
        Some(ReadHit::Lane(held)) => {
            let Some(lane) = held_lane(tables, held) else {
                return 1;
            };
            lane
        }
        Some(ReadHit::Text(text)) => {
            let (_, bank, _, state, lengths) = tables.library();
            let mut slot = TraceInputSlot::other();
            if !host::publish_string(bank, state, lengths, text, &mut slot) {
                return 1;
            }
            slot.bits()
        }
        None => return 1,
    };
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = word };
    0
}

/// What a [`HelperId::TABLE_GET_INTEGER`] family read found, taken out of the
/// table before the run's lanes are touched.
enum ReadHit {
    /// The answer itself: a number or a boolean.
    Word(i64),
    /// A table, a closure or a userdata, which answers with its lane.
    Lane(Held),
    /// A string, which answers with a new string lane.
    Text(LuaString),
}

/// The [`ReadHit`] of `found` for a read of `kind`, or `None` for a value of
/// another type.
#[inline(always)]
fn read_hit(kind: ReadKind, found: &Value) -> Option<ReadHit> {
    Some(match (kind, found) {
        (ReadKind::Integer, Value::Integer(value)) => ReadHit::Word(*value),
        (ReadKind::Float, Value::Number(value)) => ReadHit::Word(value.to_bits().cast_signed()),
        (ReadKind::Boolean, Value::Boolean(value)) => ReadHit::Word(i64::from(*value)),
        (ReadKind::Table, Value::Table(table)) => ReadHit::Lane(Held::Table(*table)),
        (ReadKind::Closure, Value::Closure(closure)) => ReadHit::Lane(Held::Closure(*closure)),
        (ReadKind::Userdata, Value::Userdata(userdata)) => ReadHit::Lane(Held::Userdata(*userdata)),
        (ReadKind::String, Value::String(text)) => ReadHit::Text(text.clone()),
        _ => return None,
    })
}

macro_rules! table_read_helper {
    ($name:ident, $kind:expr) => {
        extern "C" fn $name(
            context: usize,
            out: *mut i64,
            table_tag: i64,
            table_bits: i64,
            key_tag: i64,
            key_bits: i64,
            walk: i64,
        ) -> i64 {
            table_read(
                context,
                out,
                ((table_tag, table_bits), (key_tag, key_bits)),
                walk,
                $kind,
            )
        }
    };
}

table_read_helper!(helper_table_get_integer, ReadKind::Integer);
table_read_helper!(helper_table_get_float, ReadKind::Float);
table_read_helper!(helper_table_get_boolean, ReadKind::Boolean);
table_read_helper!(helper_table_get_table, ReadKind::Table);
table_read_helper!(helper_table_get_closure, ReadKind::Closure);
table_read_helper!(helper_table_get_string, ReadKind::String);
table_read_helper!(helper_table_get_userdata, ReadKind::Userdata);

/// [`HelperId::TABLE_ABSENT_INDEX`]: the step that ends an `ipairs` loop
/// over a table met by lane, the read of a key a table lacks, and `t[nil]`.
#[allow(unsafe_code)]
extern "C" fn helper_table_absent_index(
    context: usize,
    _out: *mut i64,
    table_tag: i64,
    table_bits: i64,
    key_tag: i64,
    key_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.table_lookup += 1;
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(table) = slot_table(TraceInputSlot::from_words(table_tag, table_bits), bank) else {
        return 1;
    };
    if !matches!(runtime.raw_metatable(&Value::Table(table)), Ok(None)) {
        return 1;
    }
    // A nil key (`t[nil]`, which the recorder types) is absent from every
    // table.
    if key_tag == TRACE_TAG_NIL {
        return 0;
    }
    let mut scratch = Value::Nil;
    let Some(key) = slot_key(
        TraceInputSlot::from_words(key_tag, key_bits),
        (bank, roots),
        &mut scratch,
    ) else {
        return 1;
    };
    match runtime.table_get_raw_value(table, key) {
        Ok(None | Some(Value::Nil)) => 0,
        _ => 1,
    }
}

/// [`HelperId::TABLE_LENGTH`]: the border of a table met by lane.
#[allow(unsafe_code)]
extern "C" fn helper_table_length(
    context: usize,
    out: *mut i64,
    tag: i64,
    bits: i64,
    raw: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.table_lookup += 1;
    let (runtime, bank, _, _, _) = tables.library();
    let Some(table) = slot_table(TraceInputSlot::from_words(tag, bits), bank) else {
        return 1;
    };
    let length = if raw != 0 {
        runtime.table_length(table).ok()
    } else {
        runtime
            .table_length_without_len_metamethod(table)
            .ok()
            .flatten()
    };
    let Some(length) = length else {
        return 1;
    };
    #[allow(clippy::cast_possible_truncation)]
    let border = length as i64;
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = border };
    0
}

/// [`HelperId::CELL_SET`]: `SetUpvalue` through the live closure. The runtime's
/// `cell_set` writes an open cell's stack slot or a closed cell's own value,
/// and tells the collector about a table or string stored into a cell it may
/// already have marked.
#[allow(unsafe_code)]
// The string and borrowed-string tags are distinct trace tag cases that
// happen to share this body.
#[allow(clippy::match_same_arms)]
extern "C" fn helper_cell_set(
    context: usize,
    _out: *mut i64,
    closure: i64,
    index: i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.closure += 1;
    let (runtime, bank, roots, _, _) = tables.library();
    let slot = TraceInputSlot::from_words(tag, bits);
    let value = match tag {
        TRACE_TAG_STRING => match host::lane_value(slot, bank, roots) {
            Some(held @ Value::String(_)) => held.clone(),
            _ => return 1,
        },
        TRACE_TAG_BORROWED_STRING => match host::lane_value(slot, bank, roots) {
            Some(held @ Value::String(_)) => held.clone(),
            _ => return 1,
        },
        TRACE_TAG_TABLE => match host::lane_value(slot, bank, roots) {
            Some(held @ Value::Table(_)) => held.clone(),
            _ => return 1,
        },
        _ => match slot.scalar_value() {
            Some(value) => value,
            None => return 1,
        },
    };
    let Some(handle) = lane_closure(bank, closure) else {
        return 1;
    };
    let Some(cell) = usize::try_from(index)
        .ok()
        .and_then(|index| runtime.closure_cell(handle, index).ok().flatten())
    else {
        return 1;
    };
    // The cell keeps its type, as native code's own write does: a cell that
    // turns into another type is the interpreter's to change, and every read
    // of the cell the trace holds an address for stays right.
    match runtime.cell_ref(cell) {
        Ok(held) if std::mem::discriminant(held) == std::mem::discriminant(&value) => {}
        _ => return 1,
    }
    i64::from(runtime.cell_set(cell, value).is_err())
}

/// [`HelperId::TABLE_INIT`]: stores a value into a table the run made, as the
/// interpreter's raw set does.
#[allow(unsafe_code)]
extern "C" fn helper_table_init(
    context: usize,
    _out: *mut i64,
    table: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(&Value::Table(handle)) = usize::try_from(table).ok().and_then(|lane| bank.get(lane))
    else {
        return 1;
    };
    // A field of a table made with a shape that names it: the slot is
    // filled in place.
    if key_tag != TRACE_TAG_INTEGER
        && key_tag != TRACE_TAG_FLOAT
        && let Some(Value::String(name)) =
            host::lane_value(TraceInputSlot::from_words(key_tag, key_bits), bank, roots)
    {
        let mut scratch = Value::Nil;
        let Some(value) = stored_value(
            TraceInputSlot::from_words(value_tag, value_bits),
            bank,
            roots,
            &mut scratch,
        ) else {
            return 1;
        };
        if runtime.table_init_shaped(handle, name, value) {
            return 0;
        }
        // Any other string field is a raw store of a key the table lacks.
        return i64::from(!matches!(
            runtime.table_set_string_from(handle, name, value),
            Ok(true)
        ));
    }
    let key = match key_tag {
        TRACE_TAG_INTEGER => super::deopt::integer_key(runtime, key_bits),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(key_bits.cast_unsigned())),
        _ => match host::lane_value(TraceInputSlot::from_words(key_tag, key_bits), bank, roots) {
            Some(Value::String(key)) => Value::String(key.clone()),
            _ => return 1,
        },
    };
    let mut scratch = Value::Nil;
    let Some(value) = stored_value(
        TraceInputSlot::from_words(value_tag, value_bits),
        bank,
        roots,
        &mut scratch,
    ) else {
        return 1;
    };
    // The array fill of a constructor: a number or boolean, or a string or
    // table, at the next key of a table with no metatable.
    let array_key = matches!(key, Value::Integer(_) | Value::Number(_));
    let stored = array_key
        && match value {
            Value::Number(_) | Value::Integer(_) | Value::Boolean(_) => {
                runtime.table_set_dense_scalar(handle, &key, value)
            }
            Value::String(_) | Value::Table(_) => {
                runtime.table_set_dense_reference(handle, &key, value)
            }
            _ => false,
        };
    if stored {
        return 0;
    }
    let value = value.clone();
    i64::from(runtime.table_set(handle, &key, value).is_err())
}

/// Whether a store under `key` may change a metatable's handler, which a
/// trace assumed at entry: the interpreter makes that store.
fn names_metamethod(key: &LuaString) -> bool {
    key.as_bytes().starts_with(b"__")
}

/// The value a slot stands for in a store: a scalar (kept in `scratch`), or
/// the string or table a lane holds. `None` for any other slot.
fn stored_value<'a>(
    slot: TraceInputSlot,
    bank: &'a [Value],
    roots: &'a [Value],
    scratch: &'a mut Value,
) -> Option<&'a Value> {
    *scratch = match slot.tag() {
        TRACE_TAG_NIL => Value::Nil,
        TRACE_TAG_BOOLEAN => Value::Boolean(slot.bits() != 0),
        TRACE_TAG_INTEGER => Value::Integer(slot.bits()),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(slot.bits().cast_unsigned())),
        _ => {
            return host::lane_value(slot, bank, roots)
                .filter(|held| matches!(held, Value::String(_) | Value::Table(_)));
        }
    };
    Some(scratch)
}

/// `table[key] = value` for a present string key and a number or boolean of
/// the type it holds (see [`HelperId::SET_PRESENT`]): the value is overwritten
/// in place. It misses, having changed nothing, for any other store. A
/// metamethod key needs no check: what a trace assumes about a metatable is
/// whether a key is present, which an overwrite keeps, and a number or boolean
/// under it is no handler a trace runs. Native code makes this store itself
/// where the table has a string index (`Lowerer::hash_store`), so the helper
/// answers only the runs and keys that have none.
#[allow(unsafe_code)]
extern "C" fn helper_set_present(
    context: usize,
    _out: *mut i64,
    table: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.set_present += 1;
    let Some(handle) = bank_table(tables, table) else {
        return 1;
    };
    let value = match value_tag {
        TRACE_TAG_INTEGER => Value::Integer(value_bits),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(value_bits.cast_unsigned())),
        TRACE_TAG_BOOLEAN => Value::Boolean(value_bits != 0),
        _ => return 1,
    };
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(Value::String(key)) =
        host::lane_value(TraceInputSlot::from_words(key_tag, key_bits), bank, roots)
    else {
        return 1;
    };
    i64::from(!matches!(
        runtime.table_overwrite_scalar_string(handle, key, &value),
        Ok(true)
    ))
}

/// `table[key] = value` for a string key the table holds as a deleted key and
/// a number or boolean (see [`HelperId::SET_REVIVE`]): the entry takes the
/// value where it stands. It misses, having changed nothing, for any other
/// store, for a key that may name a metamethod, and for a key whose value a
/// binding of the run fixes ([`RunTables::pinned_keys`]). Native code makes
/// this store itself where the table has a string index and the run allows it
/// (`Lowerer::hash_store`); the helper answers the rest.
#[allow(unsafe_code)]
extern "C" fn helper_set_revive(
    context: usize,
    _out: *mut i64,
    table: i64,
    key_tag: i64,
    key_bits: i64,
    value_tag: i64,
    value_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.set_revive += 1;
    let value = match value_tag {
        TRACE_TAG_INTEGER => Value::Integer(value_bits),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(value_bits.cast_unsigned())),
        TRACE_TAG_BOOLEAN => Value::Boolean(value_bits != 0),
        _ => return 1,
    };
    let pinned = tables.pinned_keys();
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(&Value::Table(handle)) = usize::try_from(table).ok().and_then(|lane| bank.get(lane))
    else {
        return 1;
    };
    let Some(Value::String(key)) =
        host::lane_value(TraceInputSlot::from_words(key_tag, key_bits), bank, roots)
    else {
        return 1;
    };
    if names_metamethod(key) || pinned.holds(key.as_bytes()) || pinned.holds_globals(handle) {
        return 1;
    }
    i64::from(!matches!(
        runtime.table_revive_string(handle, key, &value),
        Ok(true)
    ))
}

/// `table[key] = nil` for a string key (see [`HelperId::DELETE_STRING`]): a
/// key of a hashed table stays as a deleted entry the traversals skip, and an
/// absent key stays absent. It misses, having changed nothing, for a key
/// whose value a binding of the run fixes ([`RunTables::pinned_keys`]), a key
/// that may name a metamethod, and any other store. Native code deletes a
/// number or boolean in place itself where the table has a string index and
/// the run allows it (`Lowerer::hash_store`); the helper answers the rest.
#[allow(unsafe_code)]
extern "C" fn helper_delete_string(
    context: usize,
    _out: *mut i64,
    table: i64,
    key_tag: i64,
    key_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    tables.store_calls.delete_string += 1;
    let pinned = tables.pinned_keys();
    let (runtime, bank, roots, _, _) = tables.library();
    let Some(&Value::Table(handle)) = usize::try_from(table).ok().and_then(|lane| bank.get(lane))
    else {
        return 1;
    };
    let Some(Value::String(key)) =
        host::lane_value(TraceInputSlot::from_words(key_tag, key_bits), bank, roots)
    else {
        return 1;
    };
    if names_metamethod(key) || pinned.holds(key.as_bytes()) || pinned.holds_globals(handle) {
        return 1;
    }
    i64::from(!matches!(
        runtime.table_delete_string_in_place(handle, key),
        Ok(true)
    ))
}

/// The check before a virtual table gets its metatable (see
/// [`HelperId::PLAIN_METATABLE`]): no `__gc`, no `__mode`, and none of the
/// metamethods the mask names. It changes nothing.
#[allow(unsafe_code)]
extern "C" fn helper_plain_metatable(
    context: usize,
    _out: *mut i64,
    metatable: i64,
    mask: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, _, _, _) = tables.library();
    let Some(Value::Table(metatable)) = usize::try_from(metatable)
        .ok()
        .and_then(|lane| bank.get(lane))
    else {
        return 1;
    };
    let mut events = Metamethod::Gc.bit() | Metamethod::Mode.bit();
    if mask & 1 != 0 {
        events |= Metamethod::Index.bit();
    }
    if mask & 2 != 0 {
        events |= Metamethod::NewIndex.bit();
    }
    if mask & 4 != 0 {
        events |= Metamethod::Len.bit();
    }
    i64::from(!runtime.metatable_lacks_all(*metatable, events))
}

/// `rawequal` of two strings: whether their bytes are equal. Misses (1) for a
/// slot that holds no string.
#[allow(unsafe_code)]
extern "C" fn helper_string_equal(
    context: usize,
    out: *mut i64,
    left_tag: i64,
    left_bits: i64,
    right_tag: i64,
    right_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &*(context as *const RunTables<'static>) };
    // The bank and the roots alone: `library` would build the length table
    // too, which a comparison never reads.
    let (bank, roots) = tables.lanes();
    // The strings in place, not cloned through `host::with_values`: a scan
    // compares one borrowed element after another against the same string,
    // and the interpreter's comparison is a pointer test first.
    let left = host::text(TraceInputSlot::from_words(left_tag, left_bits), bank, roots);
    let right = host::text(
        TraceInputSlot::from_words(right_tag, right_bits),
        bank,
        roots,
    );
    let (Some(left), Some(right)) = (left, right) else {
        return 1;
    };
    let equal = left == right;
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = i64::from(equal) };
    0
}

/// The integer place a position argument of `table.insert` and
/// `table.remove` names: the interpreter's `Runtime::coerce_number` and a
/// truncation. `None` for a slot that is no number and for a number that has
/// no place (NaN, an infinity, out of range), which the interpreter raises
/// an error for.
fn position_of(slot: TraceInputSlot) -> Option<i64> {
    match slot.tag() {
        TRACE_TAG_INTEGER => Some(slot.bits()),
        TRACE_TAG_FLOAT => {
            let number = f64::from_bits(slot.bits().cast_unsigned());
            #[allow(clippy::cast_possible_truncation, reason = "bounded just before")]
            (number.is_finite() && number.abs() < 9.0e18).then(|| number.trunc() as i64)
        }
        _ => None,
    }
}

/// The table in lane `lane` of the run's bank.
fn bank_table(tables: &mut RunTables<'static>, lane: i64) -> Option<TableHandle> {
    let (_, bank, _, _, _) = tables.library();
    match usize::try_from(lane).ok().and_then(|lane| bank.get(lane)) {
        Some(Value::Table(table)) => Some(*table),
        _ => None,
    }
}

/// Whether the profile's `table.insert` (or `table.remove`) honours
/// `__index`, `__newindex` and `__len`: the profiles from Lua 5.3 on register
/// the kind that redispatches, the others the raw one.
fn honours_metamethods(runtime: &Runtime, kind: DirectNativeKind) -> bool {
    runtime.direct_native(kind).is_some()
}

/// `table.insert (t, value)` and `table.insert (t, position, value)` (see
/// [`HelperId::TABLE_INSERT`]). `count` is 2 or 3; the call's arguments after
/// the table are the position, if any, and the value.
#[allow(unsafe_code)]
extern "C" fn helper_table_insert(
    context: usize,
    _out: *mut i64,
    words: *mut TraceInputSlot,
    count: i64,
    table: i64,
    first_tag: i64,
    first_bits: i64,
    second_tag: i64,
    second_bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let first = TraceInputSlot::from_words(first_tag, first_bits);
    let (position, value) = match count {
        2 => (None, first),
        3 => (
            Some(position_of(first)),
            TraceInputSlot::from_words(second_tag, second_bits),
        ),
        _ => return 1,
    };
    let position = match position {
        None => None,
        Some(Some(place)) if place >= 1 => match usize::try_from(place) {
            Ok(place) => Some(place),
            Err(_) => return 1,
        },
        Some(_) => return 1,
    };
    let Some(handle) = bank_table(tables, table) else {
        return 1;
    };
    let (runtime, bank, roots, _, _) = tables.library();
    let honours = honours_metamethods(runtime, DirectNativeKind::TableInsert);
    let view = host::with_values(&[value], bank, roots, |values| {
        runtime.table_trace_insert(handle, position, &values[0], honours)
    })
    .flatten();
    let Some(view) = view else {
        return 1;
    };
    tables.rewrite_bindings(words, handle, view);
    0
}

/// What a `table.remove` helper does with the element it takes out.
#[derive(Clone, Copy)]
enum Removed {
    Integer,
    Float,
    Boolean,
    Table,
    String,
}

impl Removed {
    fn accepts(self, value: &Value) -> bool {
        matches!(
            (self, value),
            (Self::Integer, Value::Integer(_))
                | (Self::Float, Value::Number(_))
                | (Self::Boolean, Value::Boolean(_))
                | (Self::Table, Value::Table(_))
                | (Self::String, Value::String(_))
        )
    }
}

/// The place `table.remove`'s optional second argument names: `None` for no
/// argument. A nil argument is a miss, though the interpreter takes it as the
/// default.
fn remove_position(count: i64, slot: TraceInputSlot) -> Result<Option<i64>, ()> {
    match count {
        1 => Ok(None),
        2 => position_of(slot).map(Some).ok_or(()),
        _ => Err(()),
    }
}

/// `table.remove (t)` and `table.remove (t, position)` of a dense list (see
/// [`HelperId::TABLE_REMOVE_INTEGER`]): writes the element taken out to `out`.
#[allow(unsafe_code)]
fn table_remove(
    context: usize,
    out: *mut i64,
    words: *mut TraceInputSlot,
    (count, table): (i64, i64),
    (tag, bits): (i64, i64),
    removed: Removed,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let Ok(position) = remove_position(count, TraceInputSlot::from_words(tag, bits)) else {
        return 1;
    };
    let place = match position {
        None => None,
        Some(place) if place >= 1 => match usize::try_from(place) {
            Ok(place) => Some(place),
            Err(_) => return 1,
        },
        Some(_) => return 1,
    };
    let Some(handle) = bank_table(tables, table) else {
        return 1;
    };
    // A table element is named by a lane, which must exist before anything
    // is taken out.
    if matches!(removed, Removed::Table) && !tables.has_lane_room() {
        return 1;
    }
    let (runtime, bank, _, state, lengths) = tables.library();
    let honours = honours_metamethods(runtime, DirectNativeKind::TableRemove);
    // A string element gets its lane before anything is taken out, so a run
    // out of room for strings misses with the table as it was.
    let mut string_lane = None;
    if matches!(removed, Removed::String) {
        let Some(Value::String(text)) = runtime.table_trace_removed(handle, place, honours) else {
            return 1;
        };
        let mut slot = TraceInputSlot::other();
        if !host::publish(bank, state, lengths, Value::String(text.clone()), &mut slot) {
            return 1;
        }
        string_lane = Some(slot.bits());
    }
    let Some((element, view)) =
        runtime.table_trace_remove(handle, place, honours, |value| removed.accepts(value))
    else {
        return 1;
    };
    let word = match element {
        Value::String(_) => match string_lane {
            Some(lane) => lane,
            None => return 1,
        },
        Value::Integer(integer) => integer,
        Value::Number(number) => number.to_bits().cast_signed(),
        Value::Boolean(flag) => i64::from(flag),
        Value::Table(_) => {
            // `has_lane_room` promised one.
            let lane = tables
                .lane(&element)
                .and_then(|lane| i64::try_from(lane).ok());
            debug_assert!(lane.is_some(), "a table lane was promised");
            let Some(lane) = lane else {
                return 1;
            };
            lane
        }
        _ => return 1,
    };
    tables.rewrite_bindings(words, handle, view);
    // SAFETY: `out` is the control block's `helper_out` (see `invoke`).
    unsafe { *out = word };
    0
}

macro_rules! remove_helper {
    ($name:ident, $removed:expr) => {
        extern "C" fn $name(
            context: usize,
            out: *mut i64,
            words: *mut TraceInputSlot,
            count: i64,
            table: i64,
            tag: i64,
            bits: i64,
        ) -> i64 {
            table_remove(context, out, words, (count, table), (tag, bits), $removed)
        }
    };
}

remove_helper!(helper_table_remove_integer, Removed::Integer);
remove_helper!(helper_table_remove_float, Removed::Float);
remove_helper!(helper_table_remove_boolean, Removed::Boolean);
remove_helper!(helper_table_remove_table, Removed::Table);
remove_helper!(helper_table_remove_string, Removed::String);

/// `table.remove` where the position is outside `1..=#t` (see
/// [`HelperId::TABLE_REMOVE_NOTHING`]): the call gives nil and changes
/// nothing. Misses when the position is inside.
#[allow(unsafe_code)]
extern "C" fn helper_table_remove_nothing(
    context: usize,
    _out: *mut i64,
    count: i64,
    table: i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let Ok(position) = remove_position(count, TraceInputSlot::from_words(tag, bits)) else {
        return 1;
    };
    let Some(handle) = bank_table(tables, table) else {
        return 1;
    };
    let (runtime, _, _, _, _) = tables.library();
    let honours = honours_metamethods(runtime, DirectNativeKind::TableRemove);
    i64::from(!runtime.table_trace_remove_nothing(handle, position, honours))
}

/// How many lanes of the run's bank a `next` step needs for the key and the
/// value it gives: a table of either takes one (a string takes a lane of
/// its own budget when it is published).
fn next_lanes(key: &Value, value: &Value) -> usize {
    usize::from(matches!(key, Value::Table(_))) + usize::from(matches!(value, Value::Table(_)))
}

/// The entry `next (t, key)` gives for the table in lane `table`, or `None`
/// when the interpreter must make the call: a table that is no lane of the
/// run, a key the bank cannot name, or an invalid continuation key (the call
/// raises an error for it). The end of the traversal is the entry `(nil,
/// nil)`. It is the interpreter's own step (`Vm::for_in_builtin_step`), so
/// the keys have the profile's types and a cursor the table kept answers
/// without a lookup.
fn next_entry(
    tables: &mut RunTables<'static>,
    table: i64,
    key: TraceInputSlot,
) -> Option<(Value, Value)> {
    let handle = bank_table(tables, table)?;
    let (runtime, bank, roots, _, _) = tables.library();
    // The scalars need no work, and a string or a table of the bank is
    // a reference to count.
    let previous = match key.tag() {
        TRACE_TAG_NIL => Value::Nil,
        TRACE_TAG_BOOLEAN => Value::Boolean(key.bits() != 0),
        TRACE_TAG_INTEGER => Value::Integer(key.bits()),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(key.bits().cast_unsigned())),
        TRACE_TAG_STRING | TRACE_TAG_TABLE => host::lane_value(key, bank, roots)?.clone(),
        _ => host::with_values(&[key], bank, roots, |values| values[0].clone())?,
    };
    match runtime.table_next_fast(handle, &previous) {
        Some(pair) => Some(pair),
        None => runtime.table_next(handle, &previous).ok(),
    }
}

/// The word a result of `kind` has for `value`, or `None` when the value is
/// of another type. A table is a lane (see [`next_lanes`]) and a string a new
/// lane of the run's root bank.
fn next_word(tables: &mut RunTables<'static>, kind: NextKind, value: &Value) -> Option<i64> {
    match (kind, value) {
        (NextKind::Integer, Value::Integer(integer)) => Some(*integer),
        (NextKind::Float, Value::Number(number)) => Some(number.to_bits().cast_signed()),
        (NextKind::Boolean, Value::Boolean(flag)) => Some(i64::from(*flag)),
        (NextKind::Table, Value::Table(_)) => {
            tables.lane(value).and_then(|lane| i64::try_from(lane).ok())
        }
        (NextKind::String, Value::String(_)) => {
            let (_, bank, _, state, lengths) = tables.library();
            let mut slot = TraceInputSlot::other();
            (host::publish(bank, state, lengths, value.clone(), &mut slot)
                && slot.tag() == TRACE_TAG_STRING)
                .then(|| slot.bits())
        }
        _ => None,
    }
}

/// Whether the run has room to publish the strings among the results of a
/// step: `results` pair each result's kind with the value, and the strings
/// among them are published one after the other. A step asks before it
/// publishes anything, so a miss leaves no lane behind.
fn next_strings_fit(
    tables: &mut RunTables<'static>,
    results: [(Option<NextKind>, &Value); 2],
) -> bool {
    let mut sizes = [0_usize; 2];
    let mut count = 0;
    for (kind, value) in results {
        if let (Some(NextKind::String), Value::String(text)) = (kind, value) {
            sizes[count] = text.len();
            count += 1;
        }
    }
    count == 0 || {
        let (_, bank, _, state, _) = tables.library();
        host::strings_fit(bank, state, &sizes[..count])
    }
}

/// `next (t, key)` that gives an entry whose key is of `KEY` and whose value
/// is of `VALUE` (the kinds of [`NextKind`]; `NO_VALUE` for a call that wants
/// the key only). Misses, having changed nothing, for any other answer: the
/// kinds and the room for the lanes and strings are checked before the first
/// publication.
#[allow(unsafe_code)]
extern "C" fn helper_next<const KEY: usize, const VALUE: usize>(
    context: usize,
    out: *mut i64,
    table: i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    count_next_step();
    let Some((key, value)) = next_entry(tables, table, TraceInputSlot::from_words(tag, bits))
    else {
        return 1;
    };
    let (key_kind, value_kind) = (NextKind::ALL[KEY], NextKind::ALL.get(VALUE).copied());
    // The shapes and the lanes first, so a miss comes before anything is
    // published.
    if !next_has_kind(key_kind, &key) || value_kind.is_some_and(|kind| !next_has_kind(kind, &value))
    {
        return 1;
    }
    let wanted = next_lanes(
        &key,
        if value_kind.is_some() {
            &value
        } else {
            &Value::Nil
        },
    );
    if wanted > 0 && !tables.has_lane_room_for(wanted) {
        return 1;
    }
    if !next_strings_fit(tables, [(Some(key_kind), &key), (value_kind, &value)]) {
        return 1;
    }
    let Some(key_word) = next_word(tables, key_kind, &key) else {
        return 1;
    };
    let Some(value_kind) = value_kind else {
        write_words(out, &[key_word]);
        return 0;
    };
    let Some(value_word) = next_word(tables, value_kind, &value) else {
        return 1;
    };
    write_words(out, &[key_word, value_word]);
    0
}

/// `next (t, key)` that gives no entry (see [`HelperId::NEXT_END`]).
#[allow(unsafe_code)]
extern "C" fn helper_next_end(
    context: usize,
    _out: *mut i64,
    table: i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    match next_entry(tables, table, TraceInputSlot::from_words(tag, bits)) {
        Some((Value::Nil, _)) => 0,
        _ => 1,
    }
}

/// The entry after the key the control lane `control` holds, for the table
/// in lane `table`: the interpreter's own step (`Vm::for_in_builtin_step`),
/// answered by the table's traversal cursor without a lookup where it can.
/// `None` when the interpreter must make the call: a table that is no lane
/// of the run, or an invalid continuation key (the call raises an error for
/// it). The end of the traversal is the entry `(nil, nil)`.
fn next_box_entry(
    tables: &mut RunTables<'static>,
    table: i64,
    control: usize,
) -> Option<(Value, Value)> {
    let handle = bank_table(tables, table)?;
    let (runtime, bank, _, _, _) = tables.library();
    let previous = bank.get(control)?;
    match runtime.table_next_fast(handle, previous) {
        Some(pair) => Some(pair),
        None => runtime.table_next(handle, previous).ok(),
    }
}

/// The index of the array part a traversal key names: a whole number from 1.
fn array_key(key: &Value) -> Option<usize> {
    match key {
        Value::Integer(index) => usize::try_from(*index).ok(),
        Value::Number(index) if *index >= 1.0 && index.fract() == 0.0 =>
        {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Some(*index as usize)
        }
        _ => None,
    }
    .filter(|index| *index >= 1)
}

/// Whether the next step of a traversal of `table` that holds `control` goes
/// to a slot of the array part: the control value is a whole number from 1,
/// and a slot that holds a value follows it inside the array part. A loop
/// over a table with no hash part that steps like that is read in place (see
/// `Recorder::boxes_loop`). (An array part that ends in nil slots leaves the
/// last entry's step to the hash part.)
pub(super) fn steps_along_array(runtime: &Runtime, table: &Value, control: &Value) -> bool {
    let Value::Table(table) = table else {
        return false;
    };
    let Some(index) = array_key(control) else {
        return false;
    };
    let Ok(len) = runtime.table_array_len(*table) else {
        return false;
    };
    index < len
        && runtime
            .table_raw_array_range(*table, index + 1, len)
            .ok()
            .flatten()
            .is_some_and(|slots| slots.iter().any(|slot| !matches!(slot, Value::Nil)))
}

thread_local! {
    /// How many steps of `next` the helpers of this thread made, for tests
    /// that want to know a loop read the array part in place.
    static NEXT_STEPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many steps of `next` the trace helpers made on this thread (counted
/// in builds with debug assertions only).
pub(crate) fn next_steps() -> u64 {
    NEXT_STEPS.with(std::cell::Cell::get)
}

/// Counts a step of `next` a helper makes. Only a build with debug
/// assertions counts (the one the tests run): the count would cost a release
/// build a few instructions per step.
#[allow(
    clippy::inline_always,
    reason = "a leaf of the step helper, folded into each instance"
)]
#[inline(always)]
fn count_next_step() {
    if cfg!(debug_assertions) {
        NEXT_STEPS.with(|steps| steps.set(steps.get() + 1));
    }
}

/// The `VALUE` of the lane-keeping step that stores the value in a lane (see
/// [`HelperId::next_box`]).
const NEXT_BOX_VALUE_LANE: usize = 6;

/// Whether `value` is of the kind `kind` names.
fn next_has_kind(kind: NextKind, value: &Value) -> bool {
    matches!(
        (kind, value),
        (NextKind::Integer, Value::Integer(_))
            | (NextKind::Float, Value::Number(_))
            | (NextKind::Boolean, Value::Boolean(_))
            | (NextKind::String, Value::String(_))
            | (NextKind::Table, Value::Table(_))
    )
}

/// A step of a traversal that keeps its control value in a lane of the run's
/// bank (see [`HelperId::next_box`]): the entry after the lane's key, which
/// becomes the lane's key. It gives the key if `KEY` names a kind of
/// [`NextKind`] and the value if `VALUE` does, in that order. A `VALUE` past
/// the kinds is the helper that keeps the value in the lane after the control
/// lane instead, whatever the value is, for a loop variable the body never
/// reads.
///
/// Misses, having changed nothing, for the end of the table, for another kind
/// of key or value than the recording saw, for a bank with no room for a
/// table lane or a string, and for a continuation key the cursor cannot
/// settle. Every shape and the room are checked before anything is
/// published or stored.
#[allow(unsafe_code)]
extern "C" fn helper_next_box<const KEY: usize, const VALUE: usize>(
    context: usize,
    out: *mut i64,
    table: i64,
    control: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    count_next_step();
    let Ok(control) = usize::try_from(control) else {
        return 1;
    };
    let keeps_value = VALUE == NEXT_BOX_VALUE_LANE;
    // The lane holds the key native steps moved to. The step moves the
    // snapshot's position before it can miss, so native code steps along the
    // hash part only after a step that succeeded.
    tables.flush_cursor();
    tables.desync_cursor();
    let (runtime, bank, _, _, _) = tables.library();
    let Some(Value::Table(handle)) = usize::try_from(table).ok().and_then(|lane| bank.get(lane))
    else {
        return 1;
    };
    let handle = *handle;
    if control >= bank.len() || (keeps_value && control + 1 >= bank.len()) {
        return 1;
    }
    // The cursor hands over the entry in place: a number is read from the
    // reference, and only the key (which the lane keeps) and a value the lane
    // keeps are cloned. Every check comes before the first store.
    let tagged = runtime.tagged_integers();
    let step = runtime.table_next_ref(handle, &bank[control]);
    let hash = matches!(step, Some(NextRef::Hash(..)));
    let (key, value) = match step {
        Some(NextRef::Hash(NextKey::String(text), value)) => (BoxKey::Text(text), value),
        Some(NextRef::Hash(NextKey::Plain(key), value)) => (BoxKey::Plain(key), value),
        Some(NextRef::Array(index, value)) => (
            BoxKey::Plain(if tagged {
                Value::Integer(i64::try_from(index).unwrap_or(i64::MAX))
            } else {
                #[allow(clippy::cast_precision_loss)]
                Value::Number(index as f64)
            }),
            value,
        ),
        Some(NextRef::End) => return 1,
        // The cursor declined: the interpreter's general step.
        None => {
            return match runtime.table_next(handle, &bank[control]) {
                Ok((Value::Nil, _)) | Err(_) => 1,
                Ok(entry) => box_step_with_lanes::<KEY, VALUE>(tables, control, entry, out),
            };
        }
    };

    let key_ok = match &key {
        BoxKey::Text(_) => KEY == NEXT_NO_PART || KEY == NextKind::String as usize,
        BoxKey::Plain(key) => has_kind::<KEY>(key),
    };
    if !key_ok || !has_kind::<VALUE>(value) {
        return 1;
    }
    // A string or a table among the results needs a lane.
    if wants_lane(KEY) || wants_lane(VALUE) {
        let entry = (key.into_value(), value.clone());
        let missed = box_step_with_lanes::<KEY, VALUE>(tables, control, entry, out);
        if missed == 0 {
            tables.sync_cursor(handle, hash);
        }
        return missed;
    }
    let mut words = [0_i64; 2];
    let mut count = 0;
    if let (BoxKey::Plain(key), true) = (&key, KEY < NEXT_KINDS) {
        words[count] = word_of::<KEY>(key);
        count += 1;
    }
    if VALUE < NEXT_KINDS {
        words[count] = word_of::<VALUE>(value);
        count += 1;
    }
    write_words(out, &words[..count]);
    if keeps_value {
        bank[control + 1] = value.clone();
    }
    bank[control] = key.into_value();
    tables.sync_cursor(handle, hash);
    0
}

/// The key of a step, before the lane takes it.
enum BoxKey<'a> {
    /// The table's own string.
    Text(&'a crate::string::LuaString),
    /// Any other key, in the profile's representation.
    Plain(Value),
}

impl BoxKey<'_> {
    #[allow(
        clippy::inline_always,
        reason = "a leaf of the step helper, folded into each instance"
    )]
    #[inline(always)]
    fn into_value(self) -> Value {
        match self {
            Self::Text(text) => Value::String(text.clone()),
            Self::Plain(key) => key,
        }
    }
}

/// The `KEY` or `VALUE` of a step that gives no part.
const NEXT_NO_PART: usize = NEXT_KINDS;

/// Whether a result of the kind `KIND` takes a lane of the bank (a string or
/// a table), which the step makes where it has the bank.
const fn wants_lane(kind: usize) -> bool {
    kind == NextKind::String as usize || kind == NextKind::Table as usize
}

/// The word of a number or boolean `value` of the kind `KIND`, whose kind the
/// step has checked.
#[allow(
    clippy::inline_always,
    reason = "a leaf of the step helper, folded into each instance"
)]
#[inline(always)]
fn word_of<const KIND: usize>(value: &Value) -> i64 {
    match value {
        Value::Integer(integer) if KIND == NextKind::Integer as usize => *integer,
        Value::Number(number) if KIND == NextKind::Float as usize => number.to_bits().cast_signed(),
        Value::Boolean(flag) if KIND == NextKind::Boolean as usize => i64::from(*flag),
        _ => 0,
    }
}

/// Whether `value` is of the kind `KIND` of [`NextKind`] (any value for a
/// part the step does not give).
#[allow(
    clippy::inline_always,
    reason = "a leaf of the step helper, folded into each instance"
)]
#[inline(always)]
fn has_kind<const KIND: usize>(value: &Value) -> bool {
    match NextKind::ALL.get(KIND) {
        Some(kind) => next_has_kind(*kind, value),
        None => true,
    }
}

/// The rest of [`helper_next_box`] for an entry that has a string or a table
/// among its results: the lanes the results take, then the stores.
fn box_step_with_lanes<const KEY: usize, const VALUE: usize>(
    tables: &mut RunTables<'static>,
    control: usize,
    (key, value): (Value, Value),
    out: *mut i64,
) -> i64 {
    let (key_kind, value_kind) = (NextKind::ALL.get(KEY), NextKind::ALL.get(VALUE));
    // The end of the traversal is the end helper's.
    if matches!(key, Value::Nil)
        || key_kind.is_some_and(|kind| !next_has_kind(*kind, &key))
        || value_kind.is_some_and(|kind| !next_has_kind(*kind, &value))
    {
        return 1;
    }
    let keeps_value = VALUE == NEXT_BOX_VALUE_LANE;
    if keeps_value && tables.library().1.get(control + 1).is_none() {
        return 1;
    }
    // The lanes first, so a run out of room misses before anything is
    // published or stored.
    let wanted = usize::from(key_kind.is_some() && matches!(key, Value::Table(_)))
        + usize::from(value_kind.is_some() && matches!(value, Value::Table(_)));
    if wanted > 0 && !tables.has_lane_room_for(wanted) {
        return 1;
    }
    if !next_strings_fit(
        tables,
        [(key_kind.copied(), &key), (value_kind.copied(), &value)],
    ) {
        return 1;
    }
    let mut words = [0_i64; 2];
    let mut count = 0;
    if let Some(kind) = key_kind {
        let Some(word) = next_word(tables, *kind, &key) else {
            return 1;
        };
        words[count] = word;
        count += 1;
    }
    if let Some(kind) = value_kind {
        let Some(word) = next_word(tables, *kind, &value) else {
            return 1;
        };
        words[count] = word;
        count += 1;
    }
    let bank = tables.library().1;
    if keeps_value {
        bank[control + 1] = value;
    }
    bank[control] = key;
    write_words(out, &words[..count]);
    0
}

/// The end of a traversal that keeps its control value in a lane (see
/// [`HelperId::NEXT_BOX_END`]).
#[allow(unsafe_code)]
extern "C" fn helper_next_box_end(context: usize, _out: *mut i64, table: i64, control: i64) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let Ok(control) = usize::try_from(control) else {
        return 1;
    };
    tables.flush_cursor();
    match next_box_entry(tables, table, control) {
        Some((Value::Nil, _)) => 0,
        _ => 1,
    }
}

/// Stores a scalar in a control lane (see [`HelperId::BOX_SET`]).
#[allow(unsafe_code)]
extern "C" fn helper_box_set(
    context: usize,
    _out: *mut i64,
    control: i64,
    tag: i64,
    bits: i64,
) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let Ok(control) = usize::try_from(control) else {
        return 1;
    };
    let slot = TraceInputSlot::from_words(tag, bits);
    // The lane no longer holds the key the snapshot's position names.
    tables.drop_pending_cursor();
    tables.desync_cursor();
    let (_, bank, roots, _, _) = tables.library();
    let value = match slot.tag() {
        TRACE_TAG_NIL => Value::Nil,
        TRACE_TAG_BOOLEAN => Value::Boolean(bits != 0),
        TRACE_TAG_INTEGER => Value::Integer(bits),
        TRACE_TAG_FLOAT => Value::Number(f64::from_bits(bits.cast_unsigned())),
        TRACE_TAG_STRING | TRACE_TAG_TABLE => match host::lane_value(slot, bank, roots) {
            Some(held) => held.clone(),
            None => return 1,
        },
        _ => return 1,
    };
    match bank.get_mut(control) {
        Some(lane) => *lane = value,
        None => return 1,
    }
    0
}

/// The check before `pairs (t)` where `pairs` honours `__pairs` (see
/// [`HelperId::PAIRS_PLAIN`]): the interpreter's own test of
/// `Vm::dispatch_call_value`.
#[allow(unsafe_code)]
extern "C" fn helper_pairs_plain(context: usize, _out: *mut i64, table: i64) -> i64 {
    // SAFETY: see `invoke`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let Some(handle) = bank_table(tables, table) else {
        return 1;
    };
    let (runtime, _, _, _, _) = tables.library();
    i64::from(!pairs_is_plain(runtime, handle))
}

/// Whether `pairs (t)` returns `next, t, nil` for the table `table`: no
/// handler in the field `__pairs` of its metatable. The profiles of the Lua
/// 5.1 family never look, and a trace of one does not call this.
pub(super) fn pairs_is_plain(runtime: &Runtime, table: TableHandle) -> bool {
    match runtime.raw_metatable(&Value::Table(table)) {
        Ok(None) => true,
        Ok(Some(metatable)) => runtime
            .table_get(metatable, &Value::from("__pairs"))
            .is_ok_and(|handler| matches!(handler, Value::Nil)),
        Err(_) => false,
    }
}

/// Installs the helper of each shape of a family: `$id (args)` names the
/// shape, the function is its instance.
macro_rules! install_shaped {
    ($($id:ident ($($shape:expr),*) => $function:expr,)*) => {
        $(install_helper(HelperId::$id($($shape),*), $function as *const () as usize);)*
    };
}

/// [`install_shaped`] for the specialized twins of a family.
macro_rules! install_prepared_shaped {
    ($($id:ident ($($shape:expr),*) => $function:expr,)*) => {
        $(install_prepared_helper(HelperId::$id($($shape),*), $function as *const () as usize);)*
    };
}

/// Installs the `next` and `pairs` helpers.
fn install_next() {
    macro_rules! box_row {
        ($key:literal) => {
            box_row!($key; 0, 1, 2, 3, 4, 5, 6);
        };
        ($key:literal; $($value:literal),*) => {
            $(install_helper(
                HelperId(HelperId::NEXT_BOX_BASE + $key * NEXT_BOX_ROW + $value),
                helper_next_box::<$key, $value> as *const () as usize,
            );)*
        };
    }
    box_row!(0);
    box_row!(1);
    box_row!(2);
    box_row!(3);
    box_row!(4);
    box_row!(5);
    install_helper(
        HelperId::NEXT_BOX_END,
        helper_next_box_end as *const () as usize,
    );
    install_helper(HelperId::BOX_SET, helper_box_set as *const () as usize);
    macro_rules! row {
        ($key:literal) => {
            row!($key; 0, 1, 2, 3, 4, 5);
        };
        ($key:literal; $($value:literal),*) => {
            $(install_helper(
                HelperId(HelperId::NEXT_BASE + $key * NEXT_ROW + $value),
                helper_next::<$key, $value> as *const () as usize,
            );)*
        };
    }
    row!(0);
    row!(1);
    row!(2);
    row!(3);
    row!(4);
    install_helper(HelperId::NEXT_END, helper_next_end as *const () as usize);
    install_helper(
        HelperId::PAIRS_PLAIN,
        helper_pairs_plain as *const () as usize,
    );
}

/// Installs the library's helpers that make and fill tables again.
#[cfg(test)]
pub(super) fn restore_table_helpers() {
    for (id, address) in [
        (HelperId::TABLE_NEW, helper_table_new as *const () as usize),
        (
            HelperId::TABLE_INIT,
            helper_table_init as *const () as usize,
        ),
        (
            HelperId::SET_STRING,
            helper_set_string as *const () as usize,
        ),
    ] {
        install_helper(id, address);
    }
}

/// Installs the addresses of the library helpers with the backend.
pub(super) fn install() {
    let table: [(HelperId, usize); 69] = [
        (
            HelperId::STRING_CHAR,
            helper_string_char as *const () as usize,
        ),
        (HelperId::TOSTRING, helper_tostring as *const () as usize),
        (
            HelperId::STRING_FORMAT,
            helper_string_format as *const () as usize,
        ),
        (
            HelperId::STRING_FORMAT_LUAJIT,
            helper_string_format_luajit as *const () as usize,
        ),
        (HelperId::CONCAT, helper_concat as *const () as usize),
        (
            HelperId::STRING_FIND,
            helper_string_find as *const () as usize,
        ),
        (
            HelperId::STRING_FIND_ABSENT,
            helper_string_find_absent as *const () as usize,
        ),
        (
            HelperId::TABLE_CONCAT,
            helper_table_concat as *const () as usize,
        ),
        (
            HelperId::STRING_FORMAT_NATIVE,
            helper_string_format_native as *const () as usize,
        ),
        (
            HelperId::GETMETATABLE,
            helper_getmetatable as *const () as usize,
        ),
        (
            HelperId::GETMETATABLE_ABSENT,
            helper_getmetatable_absent as *const () as usize,
        ),
        (
            HelperId::SETMETATABLE,
            helper_setmetatable as *const () as usize,
        ),
        (
            HelperId::PLAIN_METATABLE,
            helper_plain_metatable as *const () as usize,
        ),
        (
            HelperId::STRING_EQUAL,
            helper_string_equal as *const () as usize,
        ),
        (
            HelperId::CONCAT_ERROR_TEXT,
            helper_concat_error_text as *const () as usize,
        ),
        (
            HelperId::TABLE_INSERT,
            helper_table_insert as *const () as usize,
        ),
        (
            HelperId::TABLE_REMOVE_INTEGER,
            helper_table_remove_integer as *const () as usize,
        ),
        (
            HelperId::TABLE_REMOVE_FLOAT,
            helper_table_remove_float as *const () as usize,
        ),
        (
            HelperId::TABLE_REMOVE_BOOLEAN,
            helper_table_remove_boolean as *const () as usize,
        ),
        (
            HelperId::TABLE_REMOVE_TABLE,
            helper_table_remove_table as *const () as usize,
        ),
        (
            HelperId::TABLE_REMOVE_NOTHING,
            helper_table_remove_nothing as *const () as usize,
        ),
        (
            HelperId::TABLE_REMOVE_STRING,
            helper_table_remove_string as *const () as usize,
        ),
        (
            HelperId::STRING_SUB,
            helper_string_sub as *const () as usize,
        ),
        (
            HelperId::STRING_SLICE,
            helper_string_sub as *const () as usize,
        ),
        (
            HelperId::STRING_REP,
            helper_string_rep as *const () as usize,
        ),
        (
            HelperId::STRING_UPPER,
            helper_string_upper as *const () as usize,
        ),
        (
            HelperId::STRING_LOWER,
            helper_string_lower as *const () as usize,
        ),
        (
            HelperId::STRING_REVERSE,
            helper_string_reverse as *const () as usize,
        ),
        (
            HelperId::MATH_RANDOM_FLOAT,
            helper_math_random_float as *const () as usize,
        ),
        (
            HelperId::MATH_RANDOM_INTEGER,
            helper_math_random_integer as *const () as usize,
        ),
        (
            HelperId::MATH_MODF_INTEGER,
            helper_math_modf_integer as *const () as usize,
        ),
        (
            HelperId::MATH_MODF_FLOAT,
            helper_math_modf_float as *const () as usize,
        ),
        (
            HelperId::TONUMBER_INTEGER,
            helper_tonumber_integer as *const () as usize,
        ),
        (
            HelperId::TONUMBER_FLOAT,
            helper_tonumber_float as *const () as usize,
        ),
        (
            HelperId::TONUMBER_NIL,
            helper_tonumber_nil as *const () as usize,
        ),
        (
            HelperId::STRING_BYTE,
            helper_string_byte as *const () as usize,
        ),
        (
            HelperId::STRING_METHOD,
            helper_string_method as *const () as usize,
        ),
        (
            HelperId::CALL_HANDLER,
            helper_call_handler as *const () as usize,
        ),
        (
            HelperId::SET_STRING,
            helper_set_string as *const () as usize,
        ),
        (
            HelperId::SET_PRESENT,
            helper_set_present as *const () as usize,
        ),
        (
            HelperId::DELETE_STRING,
            helper_delete_string as *const () as usize,
        ),
        (
            HelperId::SET_REVIVE,
            helper_set_revive as *const () as usize,
        ),
        (
            HelperId::INDEX_HANDLER,
            helper_index_handler as *const () as usize,
        ),
        (
            HelperId::NEWINDEX_HANDLER,
            helper_newindex_handler as *const () as usize,
        ),
        (
            HelperId::LEN_HANDLER,
            helper_len_handler as *const () as usize,
        ),
        (
            HelperId::TABLE_SET_STRING,
            helper_table_set_string as *const () as usize,
        ),
        (
            HelperId::OPERATOR_HANDLER,
            helper_operator_handler as *const () as usize,
        ),
        (HelperId::TABLE_NEW, helper_table_new as *const () as usize),
        (
            HelperId::TABLE_INIT,
            helper_table_init as *const () as usize,
        ),
        (
            HelperId::CLOSURE_SAME_PROTO,
            helper_closure_same_proto as *const () as usize,
        ),
        (
            HelperId::CELL_GET_INTEGER,
            helper_cell_get_integer as *const () as usize,
        ),
        (
            HelperId::CELL_GET_FLOAT,
            helper_cell_get_float as *const () as usize,
        ),
        (
            HelperId::CELL_GET_BOOLEAN,
            helper_cell_get_boolean as *const () as usize,
        ),
        (
            HelperId::CELL_GET_TABLE,
            helper_cell_get_table as *const () as usize,
        ),
        (HelperId::CELL_SET, helper_cell_set as *const () as usize),
        (
            HelperId::TABLE_GET_INTEGER,
            helper_table_get_integer as *const () as usize,
        ),
        (
            HelperId::TABLE_GET_FLOAT,
            helper_table_get_float as *const () as usize,
        ),
        (
            HelperId::TABLE_GET_BOOLEAN,
            helper_table_get_boolean as *const () as usize,
        ),
        (
            HelperId::TABLE_GET_TABLE,
            helper_table_get_table as *const () as usize,
        ),
        (
            HelperId::TABLE_GET_CLOSURE,
            helper_table_get_closure as *const () as usize,
        ),
        (
            HelperId::TABLE_GET_STRING,
            helper_table_get_string as *const () as usize,
        ),
        (
            HelperId::TABLE_LENGTH,
            helper_table_length as *const () as usize,
        ),
        (
            HelperId::TABLE_SET_INDEX,
            helper_table_set_index as *const () as usize,
        ),
        (
            HelperId::TABLE_GET_USERDATA,
            helper_table_get_userdata as *const () as usize,
        ),
        (
            HelperId::TABLE_ABSENT_INDEX,
            helper_table_absent_index as *const () as usize,
        ),
        (
            HelperId::CELL_GET_CALLABLE,
            helper_cell_get_callable as *const () as usize,
        ),
        (
            HelperId::CELL_GET_STRING,
            helper_cell_get_string as *const () as usize,
        ),
        (
            HelperId::CLOSURE_SAME_ENV,
            helper_closure_same_env as *const () as usize,
        ),
        (
            HelperId::TABLE_SAME,
            helper_table_same as *const () as usize,
        ),
    ];
    for (id, address) in table {
        install_helper(id, address);
    }
    install_helper(
        HelperId::STRING_MATCH_ABSENT,
        helper_match_absent as *const () as usize,
    );
    install_helper(
        HelperId::GMATCH_END,
        helper_gmatch_end as *const () as usize,
    );
    install_helper(
        HelperId::GMATCH_NEW,
        helper_gmatch_new as *const () as usize,
    );
    install_helper(HelperId::STRING_GSUB, helper_gsub as *const () as usize);
    text::install();
    resume::install();
    vmcall::install();
    install_next();
    install_helper(
        HelperId::STRING_NUMBER_INTEGER,
        helper_string_number::<true> as *const () as usize,
    );
    install_helper(
        HelperId::STRING_NUMBER_FLOAT,
        helper_string_number::<false> as *const () as usize,
    );
    install_shaped! {
        string_find_pattern(0, 0) => helper_find_pattern::<0, 0, false>,
        string_find_pattern(1, 0) => helper_find_pattern::<1, 0, false>,
        string_find_pattern(1, 1) => helper_find_pattern::<1, 1, false>,
        string_find_pattern(2, 0) => helper_find_pattern::<2, 0, false>,
        string_find_pattern(2, 1) => helper_find_pattern::<2, 1, false>,
        string_find_pattern(2, 2) => helper_find_pattern::<2, 2, false>,
        string_find_pattern(2, 3) => helper_find_pattern::<2, 3, false>,
        string_match(1, 0) => helper_match::<1, 0, false>,
        gmatch_next(1, 0) => helper_gmatch_next::<1, 0, false>,
        string_match(1, 1) => helper_match::<1, 1, false>,
        gmatch_next(1, 1) => helper_gmatch_next::<1, 1, false>,
        string_match(2, 0) => helper_match::<2, 0, false>,
        gmatch_next(2, 0) => helper_gmatch_next::<2, 0, false>,
        string_match(2, 1) => helper_match::<2, 1, false>,
        gmatch_next(2, 1) => helper_gmatch_next::<2, 1, false>,
        string_match(2, 2) => helper_match::<2, 2, false>,
        gmatch_next(2, 2) => helper_gmatch_next::<2, 2, false>,
        string_match(2, 3) => helper_match::<2, 3, false>,
        gmatch_next(2, 3) => helper_gmatch_next::<2, 3, false>,
        string_match(3, 0) => helper_match::<3, 0, false>,
        gmatch_next(3, 0) => helper_gmatch_next::<3, 0, false>,
        string_match(3, 1) => helper_match::<3, 1, false>,
        gmatch_next(3, 1) => helper_gmatch_next::<3, 1, false>,
        string_match(3, 2) => helper_match::<3, 2, false>,
        gmatch_next(3, 2) => helper_gmatch_next::<3, 2, false>,
        string_match(3, 3) => helper_match::<3, 3, false>,
        gmatch_next(3, 3) => helper_gmatch_next::<3, 3, false>,
        string_match(3, 4) => helper_match::<3, 4, false>,
        gmatch_next(3, 4) => helper_gmatch_next::<3, 4, false>,
        string_match(3, 5) => helper_match::<3, 5, false>,
        gmatch_next(3, 5) => helper_gmatch_next::<3, 5, false>,
        string_match(3, 6) => helper_match::<3, 6, false>,
        gmatch_next(3, 6) => helper_gmatch_next::<3, 6, false>,
        string_match(3, 7) => helper_match::<3, 7, false>,
        gmatch_next(3, 7) => helper_gmatch_next::<3, 7, false>,
        string_match(4, 0) => helper_match::<4, 0, false>,
        gmatch_next(4, 0) => helper_gmatch_next::<4, 0, false>,
        string_match(4, 1) => helper_match::<4, 1, false>,
        gmatch_next(4, 1) => helper_gmatch_next::<4, 1, false>,
        string_match(4, 2) => helper_match::<4, 2, false>,
        gmatch_next(4, 2) => helper_gmatch_next::<4, 2, false>,
        string_match(4, 3) => helper_match::<4, 3, false>,
        gmatch_next(4, 3) => helper_gmatch_next::<4, 3, false>,
        string_match(4, 4) => helper_match::<4, 4, false>,
        gmatch_next(4, 4) => helper_gmatch_next::<4, 4, false>,
        string_match(4, 5) => helper_match::<4, 5, false>,
        gmatch_next(4, 5) => helper_gmatch_next::<4, 5, false>,
        string_match(4, 6) => helper_match::<4, 6, false>,
        gmatch_next(4, 6) => helper_gmatch_next::<4, 6, false>,
        string_match(4, 7) => helper_match::<4, 7, false>,
        gmatch_next(4, 7) => helper_gmatch_next::<4, 7, false>,
        string_match(4, 8) => helper_match::<4, 8, false>,
        gmatch_next(4, 8) => helper_gmatch_next::<4, 8, false>,
        string_match(4, 9) => helper_match::<4, 9, false>,
        gmatch_next(4, 9) => helper_gmatch_next::<4, 9, false>,
        string_match(4, 10) => helper_match::<4, 10, false>,
        gmatch_next(4, 10) => helper_gmatch_next::<4, 10, false>,
        string_match(4, 11) => helper_match::<4, 11, false>,
        gmatch_next(4, 11) => helper_gmatch_next::<4, 11, false>,
        string_match(4, 12) => helper_match::<4, 12, false>,
        gmatch_next(4, 12) => helper_gmatch_next::<4, 12, false>,
        string_match(4, 13) => helper_match::<4, 13, false>,
        gmatch_next(4, 13) => helper_gmatch_next::<4, 13, false>,
        string_match(4, 14) => helper_match::<4, 14, false>,
        gmatch_next(4, 14) => helper_gmatch_next::<4, 14, false>,
        string_match(4, 15) => helper_match::<4, 15, false>,
        gmatch_next(4, 15) => helper_gmatch_next::<4, 15, false>,
        string_find_prefix(0, 0) => helper_find_pattern::<0, 0, true>,
        string_find_prefix(1, 0) => helper_find_pattern::<1, 0, true>,
        string_find_prefix(1, 1) => helper_find_pattern::<1, 1, true>,
        string_find_prefix(2, 0) => helper_find_pattern::<2, 0, true>,
        string_find_prefix(2, 1) => helper_find_pattern::<2, 1, true>,
        string_find_prefix(2, 2) => helper_find_pattern::<2, 2, true>,
        string_find_prefix(2, 3) => helper_find_pattern::<2, 3, true>,
        string_match_prefix(1, 0) => helper_match::<1, 0, true>,
        gmatch_next_prefix(1, 0) => helper_gmatch_next::<1, 0, true>,
        string_match_prefix(1, 1) => helper_match::<1, 1, true>,
        gmatch_next_prefix(1, 1) => helper_gmatch_next::<1, 1, true>,
        string_match_prefix(2, 0) => helper_match::<2, 0, true>,
        gmatch_next_prefix(2, 0) => helper_gmatch_next::<2, 0, true>,
        string_match_prefix(2, 1) => helper_match::<2, 1, true>,
        gmatch_next_prefix(2, 1) => helper_gmatch_next::<2, 1, true>,
        string_match_prefix(2, 2) => helper_match::<2, 2, true>,
        gmatch_next_prefix(2, 2) => helper_gmatch_next::<2, 2, true>,
        string_match_prefix(2, 3) => helper_match::<2, 3, true>,
        gmatch_next_prefix(2, 3) => helper_gmatch_next::<2, 3, true>,
        string_match_prefix(3, 0) => helper_match::<3, 0, true>,
        gmatch_next_prefix(3, 0) => helper_gmatch_next::<3, 0, true>,
        string_match_prefix(3, 1) => helper_match::<3, 1, true>,
        gmatch_next_prefix(3, 1) => helper_gmatch_next::<3, 1, true>,
        string_match_prefix(3, 2) => helper_match::<3, 2, true>,
        gmatch_next_prefix(3, 2) => helper_gmatch_next::<3, 2, true>,
        string_match_prefix(3, 3) => helper_match::<3, 3, true>,
        gmatch_next_prefix(3, 3) => helper_gmatch_next::<3, 3, true>,
        string_match_prefix(3, 4) => helper_match::<3, 4, true>,
        gmatch_next_prefix(3, 4) => helper_gmatch_next::<3, 4, true>,
        string_match_prefix(3, 5) => helper_match::<3, 5, true>,
        gmatch_next_prefix(3, 5) => helper_gmatch_next::<3, 5, true>,
        string_match_prefix(3, 6) => helper_match::<3, 6, true>,
        gmatch_next_prefix(3, 6) => helper_gmatch_next::<3, 6, true>,
        string_match_prefix(3, 7) => helper_match::<3, 7, true>,
        gmatch_next_prefix(3, 7) => helper_gmatch_next::<3, 7, true>,
        string_match_prefix(4, 0) => helper_match::<4, 0, true>,
        gmatch_next_prefix(4, 0) => helper_gmatch_next::<4, 0, true>,
        string_match_prefix(4, 1) => helper_match::<4, 1, true>,
        gmatch_next_prefix(4, 1) => helper_gmatch_next::<4, 1, true>,
        string_match_prefix(4, 2) => helper_match::<4, 2, true>,
        gmatch_next_prefix(4, 2) => helper_gmatch_next::<4, 2, true>,
        string_match_prefix(4, 3) => helper_match::<4, 3, true>,
        gmatch_next_prefix(4, 3) => helper_gmatch_next::<4, 3, true>,
        string_match_prefix(4, 4) => helper_match::<4, 4, true>,
        gmatch_next_prefix(4, 4) => helper_gmatch_next::<4, 4, true>,
        string_match_prefix(4, 5) => helper_match::<4, 5, true>,
        gmatch_next_prefix(4, 5) => helper_gmatch_next::<4, 5, true>,
        string_match_prefix(4, 6) => helper_match::<4, 6, true>,
        gmatch_next_prefix(4, 6) => helper_gmatch_next::<4, 6, true>,
        string_match_prefix(4, 7) => helper_match::<4, 7, true>,
        gmatch_next_prefix(4, 7) => helper_gmatch_next::<4, 7, true>,
        string_match_prefix(4, 8) => helper_match::<4, 8, true>,
        gmatch_next_prefix(4, 8) => helper_gmatch_next::<4, 8, true>,
        string_match_prefix(4, 9) => helper_match::<4, 9, true>,
        gmatch_next_prefix(4, 9) => helper_gmatch_next::<4, 9, true>,
        string_match_prefix(4, 10) => helper_match::<4, 10, true>,
        gmatch_next_prefix(4, 10) => helper_gmatch_next::<4, 10, true>,
        string_match_prefix(4, 11) => helper_match::<4, 11, true>,
        gmatch_next_prefix(4, 11) => helper_gmatch_next::<4, 11, true>,
        string_match_prefix(4, 12) => helper_match::<4, 12, true>,
        gmatch_next_prefix(4, 12) => helper_gmatch_next::<4, 12, true>,
        string_match_prefix(4, 13) => helper_match::<4, 13, true>,
        gmatch_next_prefix(4, 13) => helper_gmatch_next::<4, 13, true>,
        string_match_prefix(4, 14) => helper_match::<4, 14, true>,
        gmatch_next_prefix(4, 14) => helper_gmatch_next::<4, 14, true>,
        string_match_prefix(4, 15) => helper_match::<4, 15, true>,
        gmatch_next_prefix(4, 15) => helper_gmatch_next::<4, 15, true>,
    }
    install_prepared_shaped! {
        gmatch_next(1, 0) => helper_gmatch_next_prepared::<1, 0, false>,
        gmatch_next(1, 1) => helper_gmatch_next_prepared::<1, 1, false>,
        gmatch_next(2, 0) => helper_gmatch_next_prepared::<2, 0, false>,
        gmatch_next(2, 1) => helper_gmatch_next_prepared::<2, 1, false>,
        gmatch_next(2, 2) => helper_gmatch_next_prepared::<2, 2, false>,
        gmatch_next(2, 3) => helper_gmatch_next_prepared::<2, 3, false>,
        gmatch_next(3, 0) => helper_gmatch_next_prepared::<3, 0, false>,
        gmatch_next(3, 1) => helper_gmatch_next_prepared::<3, 1, false>,
        gmatch_next(3, 2) => helper_gmatch_next_prepared::<3, 2, false>,
        gmatch_next(3, 3) => helper_gmatch_next_prepared::<3, 3, false>,
        gmatch_next(3, 4) => helper_gmatch_next_prepared::<3, 4, false>,
        gmatch_next(3, 5) => helper_gmatch_next_prepared::<3, 5, false>,
        gmatch_next(3, 6) => helper_gmatch_next_prepared::<3, 6, false>,
        gmatch_next(3, 7) => helper_gmatch_next_prepared::<3, 7, false>,
        gmatch_next(4, 0) => helper_gmatch_next_prepared::<4, 0, false>,
        gmatch_next(4, 1) => helper_gmatch_next_prepared::<4, 1, false>,
        gmatch_next(4, 2) => helper_gmatch_next_prepared::<4, 2, false>,
        gmatch_next(4, 3) => helper_gmatch_next_prepared::<4, 3, false>,
        gmatch_next(4, 4) => helper_gmatch_next_prepared::<4, 4, false>,
        gmatch_next(4, 5) => helper_gmatch_next_prepared::<4, 5, false>,
        gmatch_next(4, 6) => helper_gmatch_next_prepared::<4, 6, false>,
        gmatch_next(4, 7) => helper_gmatch_next_prepared::<4, 7, false>,
        gmatch_next(4, 8) => helper_gmatch_next_prepared::<4, 8, false>,
        gmatch_next(4, 9) => helper_gmatch_next_prepared::<4, 9, false>,
        gmatch_next(4, 10) => helper_gmatch_next_prepared::<4, 10, false>,
        gmatch_next(4, 11) => helper_gmatch_next_prepared::<4, 11, false>,
        gmatch_next(4, 12) => helper_gmatch_next_prepared::<4, 12, false>,
        gmatch_next(4, 13) => helper_gmatch_next_prepared::<4, 13, false>,
        gmatch_next(4, 14) => helper_gmatch_next_prepared::<4, 14, false>,
        gmatch_next(4, 15) => helper_gmatch_next_prepared::<4, 15, false>,
        gmatch_next_prefix(1, 0) => helper_gmatch_next_prepared::<1, 0, true>,
        gmatch_next_prefix(1, 1) => helper_gmatch_next_prepared::<1, 1, true>,
        gmatch_next_prefix(2, 0) => helper_gmatch_next_prepared::<2, 0, true>,
        gmatch_next_prefix(2, 1) => helper_gmatch_next_prepared::<2, 1, true>,
        gmatch_next_prefix(2, 2) => helper_gmatch_next_prepared::<2, 2, true>,
        gmatch_next_prefix(2, 3) => helper_gmatch_next_prepared::<2, 3, true>,
        gmatch_next_prefix(3, 0) => helper_gmatch_next_prepared::<3, 0, true>,
        gmatch_next_prefix(3, 1) => helper_gmatch_next_prepared::<3, 1, true>,
        gmatch_next_prefix(3, 2) => helper_gmatch_next_prepared::<3, 2, true>,
        gmatch_next_prefix(3, 3) => helper_gmatch_next_prepared::<3, 3, true>,
        gmatch_next_prefix(3, 4) => helper_gmatch_next_prepared::<3, 4, true>,
        gmatch_next_prefix(3, 5) => helper_gmatch_next_prepared::<3, 5, true>,
        gmatch_next_prefix(3, 6) => helper_gmatch_next_prepared::<3, 6, true>,
        gmatch_next_prefix(3, 7) => helper_gmatch_next_prepared::<3, 7, true>,
        gmatch_next_prefix(4, 0) => helper_gmatch_next_prepared::<4, 0, true>,
        gmatch_next_prefix(4, 1) => helper_gmatch_next_prepared::<4, 1, true>,
        gmatch_next_prefix(4, 2) => helper_gmatch_next_prepared::<4, 2, true>,
        gmatch_next_prefix(4, 3) => helper_gmatch_next_prepared::<4, 3, true>,
        gmatch_next_prefix(4, 4) => helper_gmatch_next_prepared::<4, 4, true>,
        gmatch_next_prefix(4, 5) => helper_gmatch_next_prepared::<4, 5, true>,
        gmatch_next_prefix(4, 6) => helper_gmatch_next_prepared::<4, 6, true>,
        gmatch_next_prefix(4, 7) => helper_gmatch_next_prepared::<4, 7, true>,
        gmatch_next_prefix(4, 8) => helper_gmatch_next_prepared::<4, 8, true>,
        gmatch_next_prefix(4, 9) => helper_gmatch_next_prepared::<4, 9, true>,
        gmatch_next_prefix(4, 10) => helper_gmatch_next_prepared::<4, 10, true>,
        gmatch_next_prefix(4, 11) => helper_gmatch_next_prepared::<4, 11, true>,
        gmatch_next_prefix(4, 12) => helper_gmatch_next_prepared::<4, 12, true>,
        gmatch_next_prefix(4, 13) => helper_gmatch_next_prepared::<4, 13, true>,
        gmatch_next_prefix(4, 14) => helper_gmatch_next_prepared::<4, 14, true>,
        gmatch_next_prefix(4, 15) => helper_gmatch_next_prepared::<4, 15, true>,
        string_find_pattern(0, 0) => helper_find_pattern_prepared::<0, 0, false>,
        string_find_pattern(1, 0) => helper_find_pattern_prepared::<1, 0, false>,
        string_find_pattern(1, 1) => helper_find_pattern_prepared::<1, 1, false>,
        string_find_pattern(2, 0) => helper_find_pattern_prepared::<2, 0, false>,
        string_find_pattern(2, 1) => helper_find_pattern_prepared::<2, 1, false>,
        string_find_pattern(2, 2) => helper_find_pattern_prepared::<2, 2, false>,
        string_find_pattern(2, 3) => helper_find_pattern_prepared::<2, 3, false>,
        string_match(1, 0) => helper_match_prepared::<1, 0, false>,
        string_match(1, 1) => helper_match_prepared::<1, 1, false>,
        string_match(2, 0) => helper_match_prepared::<2, 0, false>,
        string_match(2, 1) => helper_match_prepared::<2, 1, false>,
        string_match(2, 2) => helper_match_prepared::<2, 2, false>,
        string_match(2, 3) => helper_match_prepared::<2, 3, false>,
        string_match(3, 0) => helper_match_prepared::<3, 0, false>,
        string_match(3, 1) => helper_match_prepared::<3, 1, false>,
        string_match(3, 2) => helper_match_prepared::<3, 2, false>,
        string_match(3, 3) => helper_match_prepared::<3, 3, false>,
        string_match(3, 4) => helper_match_prepared::<3, 4, false>,
        string_match(3, 5) => helper_match_prepared::<3, 5, false>,
        string_match(3, 6) => helper_match_prepared::<3, 6, false>,
        string_match(3, 7) => helper_match_prepared::<3, 7, false>,
        string_match(4, 0) => helper_match_prepared::<4, 0, false>,
        string_match(4, 1) => helper_match_prepared::<4, 1, false>,
        string_match(4, 2) => helper_match_prepared::<4, 2, false>,
        string_match(4, 3) => helper_match_prepared::<4, 3, false>,
        string_match(4, 4) => helper_match_prepared::<4, 4, false>,
        string_match(4, 5) => helper_match_prepared::<4, 5, false>,
        string_match(4, 6) => helper_match_prepared::<4, 6, false>,
        string_match(4, 7) => helper_match_prepared::<4, 7, false>,
        string_match(4, 8) => helper_match_prepared::<4, 8, false>,
        string_match(4, 9) => helper_match_prepared::<4, 9, false>,
        string_match(4, 10) => helper_match_prepared::<4, 10, false>,
        string_match(4, 11) => helper_match_prepared::<4, 11, false>,
        string_match(4, 12) => helper_match_prepared::<4, 12, false>,
        string_match(4, 13) => helper_match_prepared::<4, 13, false>,
        string_match(4, 14) => helper_match_prepared::<4, 14, false>,
        string_match(4, 15) => helper_match_prepared::<4, 15, false>,
        string_find_prefix(0, 0) => helper_find_pattern_prepared::<0, 0, true>,
        string_find_prefix(1, 0) => helper_find_pattern_prepared::<1, 0, true>,
        string_find_prefix(1, 1) => helper_find_pattern_prepared::<1, 1, true>,
        string_find_prefix(2, 0) => helper_find_pattern_prepared::<2, 0, true>,
        string_find_prefix(2, 1) => helper_find_pattern_prepared::<2, 1, true>,
        string_find_prefix(2, 2) => helper_find_pattern_prepared::<2, 2, true>,
        string_find_prefix(2, 3) => helper_find_pattern_prepared::<2, 3, true>,
        string_match_prefix(1, 0) => helper_match_prepared::<1, 0, true>,
        string_match_prefix(1, 1) => helper_match_prepared::<1, 1, true>,
        string_match_prefix(2, 0) => helper_match_prepared::<2, 0, true>,
        string_match_prefix(2, 1) => helper_match_prepared::<2, 1, true>,
        string_match_prefix(2, 2) => helper_match_prepared::<2, 2, true>,
        string_match_prefix(2, 3) => helper_match_prepared::<2, 3, true>,
        string_match_prefix(3, 0) => helper_match_prepared::<3, 0, true>,
        string_match_prefix(3, 1) => helper_match_prepared::<3, 1, true>,
        string_match_prefix(3, 2) => helper_match_prepared::<3, 2, true>,
        string_match_prefix(3, 3) => helper_match_prepared::<3, 3, true>,
        string_match_prefix(3, 4) => helper_match_prepared::<3, 4, true>,
        string_match_prefix(3, 5) => helper_match_prepared::<3, 5, true>,
        string_match_prefix(3, 6) => helper_match_prepared::<3, 6, true>,
        string_match_prefix(3, 7) => helper_match_prepared::<3, 7, true>,
        string_match_prefix(4, 0) => helper_match_prepared::<4, 0, true>,
        string_match_prefix(4, 1) => helper_match_prepared::<4, 1, true>,
        string_match_prefix(4, 2) => helper_match_prepared::<4, 2, true>,
        string_match_prefix(4, 3) => helper_match_prepared::<4, 3, true>,
        string_match_prefix(4, 4) => helper_match_prepared::<4, 4, true>,
        string_match_prefix(4, 5) => helper_match_prepared::<4, 5, true>,
        string_match_prefix(4, 6) => helper_match_prepared::<4, 6, true>,
        string_match_prefix(4, 7) => helper_match_prepared::<4, 7, true>,
        string_match_prefix(4, 8) => helper_match_prepared::<4, 8, true>,
        string_match_prefix(4, 9) => helper_match_prepared::<4, 9, true>,
        string_match_prefix(4, 10) => helper_match_prepared::<4, 10, true>,
        string_match_prefix(4, 11) => helper_match_prepared::<4, 11, true>,
        string_match_prefix(4, 12) => helper_match_prepared::<4, 12, true>,
        string_match_prefix(4, 13) => helper_match_prepared::<4, 13, true>,
        string_match_prefix(4, 14) => helper_match_prepared::<4, 14, true>,
        string_match_prefix(4, 15) => helper_match_prepared::<4, 15, true>,
    }
    install_prepared_helper(
        HelperId::STRING_FORMAT,
        helper_string_format_prepared as *const () as usize,
    );
    install_prepared_helper(
        HelperId::STRING_FORMAT_LUAJIT,
        helper_string_format_luajit_prepared as *const () as usize,
    );
    install_prepared_helper(
        HelperId::STRING_FORMAT_NATIVE,
        helper_string_format_native_prepared as *const () as usize,
    );
    install_prepared_helper(
        HelperId::GMATCH_END,
        helper_gmatch_end_prepared as *const () as usize,
    );
    install_prepared_helper(
        HelperId::STRING_GSUB,
        helper_gsub_prepared as *const () as usize,
    );
    install_prepared_helper(
        HelperId::STRING_MATCH_ABSENT,
        helper_match_absent_prepared as *const () as usize,
    );
    install_prepared_helper(
        HelperId::STRING_FIND_ABSENT,
        helper_find_absent_prepared as *const () as usize,
    );
}
