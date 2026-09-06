//! The helpers of the `utf8` library and of `string.pack`, `string.unpack`
//! and `string.packsize`. The work is the natives' own code (see
//! [`super::host::text`]).

use super::super::host::text::{self, Failure};
use super::super::native::RunTables;
use super::write_words;
use crate::jit::{TraceInputSlot, install_helper};
use crate::jit_ir::trace::{HelperId, MAX_UNPACK_VALUES, MAX_UTF8_CODEPOINTS};
use crate::runtime::DirectNativeKind;
use crate::string::utf8::{CodesStep, Len, Offset};

/// Defines the helper `$name`: `extern "C" fn(context, out, count, (tag, bits)
/// x N) -> status` for a variadic helper of N values. The body gets the run's
/// tables, the arguments given and the result block.
macro_rules! variadic {
    ($name:ident, [$($tag:ident $bits:ident),+], |$tables:ident, $args:ident, $out:ident| $body:expr) => {
        #[allow(unsafe_code, clippy::too_many_arguments)]
        extern "C" fn $name(
            context: usize,
            $out: *mut i64,
            count: i64,
            $($tag: i64, $bits: i64),+
        ) -> i64 {
            let all = [$(TraceInputSlot::from_words($tag, $bits)),+];
            let $args = &all[..usize::try_from(count).unwrap_or(0).min(all.len())];
            // SAFETY: the context is the `RunTables` that handed out the host
            // entry, which outlives the run and is used by nothing else while
            // native code calls here.
            let $tables = unsafe { &mut *(context as *mut RunTables<'static>) };
            i64::from(!$body)
        }
    };
}

variadic!(
    helper_utf8_char,
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4],
    |tables, args, out| {
        let (runtime, bank, roots, state, lengths) = tables.library();
        let mut slot = TraceInputSlot::other();
        let answered = text::char_fast(runtime, bank, (state, lengths), args, &mut slot)
            .unwrap_or_else(|| {
                text::string_result(
                    runtime,
                    bank,
                    roots,
                    (state, lengths),
                    DirectNativeKind::Utf8Char,
                    args,
                    &mut slot,
                )
            });
        if answered {
            write_words(out, &[slot.bits()]);
        }
        answered
    }
);

variadic!(
    helper_string_pack,
    [t0 b0, t1 b1, t2 b2, t3 b3, t4 b4, t5 b5],
    |tables, args, out| {
        let (runtime, bank, roots, state, lengths) = tables.library();
        let mut slot = TraceInputSlot::other();
        let answered = text::pack(runtime, bank, roots, (state, lengths), args, &mut slot);
        if answered {
            write_words(out, &[slot.bits()]);
        }
        answered
    }
);

// `utf8.len` that counts the characters.
variadic!(
    helper_utf8_len,
    [t0 b0, t1 b1, t2 b2, t3 b3],
    |tables, args, out| {
        let (runtime, bank, roots, _, _) = tables.library();
        let mut words = [0_i64; 1];
        let answered = match text::len_fast(runtime, (bank, roots), args) {
            Some(Len::Count(count)) => {
                words[0] = count;
                true
            }
            Some(Len::Invalid(_)) => false,
            None => text::integers(
                runtime,
                (bank, roots),
                DirectNativeKind::Utf8Len,
                args,
                &mut words,
            ),
        };
        if answered {
            write_words(out, &words);
        }
        answered
    }
);

/// Defines the helper `$name` of `utf8.offset` that finds the character: it
/// gives `$words` positions, the start and with two the end too (Lua 5.5).
macro_rules! offset_helper {
    ($name:ident, $words:expr) => {
        variadic!(
            $name,
            [t0 b0, t1 b1, t2 b2],
            |tables, args, out| {
                let (runtime, bank, roots, _, _) = tables.library();
                let mut words = [0_i64; 2];
                let wanted = $words;
                let answered = match text::offset_fast((bank, roots), args) {
                    Some(Offset::Found { start, end }) => {
                        match (i64::try_from(start), i64::try_from(end)) {
                            (Ok(start), Ok(end)) => {
                                words = [start, end];
                                true
                            }
                            _ => false,
                        }
                    }
                    Some(Offset::Absent) => false,
                    None => text::integers(
                        runtime,
                        (bank, roots),
                        DirectNativeKind::Utf8Offset,
                        args,
                        &mut words[..wanted],
                    ),
                };
                if answered {
                    write_words(out, &words[..wanted]);
                }
                answered
            }
        );
    };
}

offset_helper!(helper_utf8_offset, 1);
offset_helper!(helper_utf8_offset_end, 2);

/// `string.packsize (format)`: the size.
#[allow(unsafe_code)]
extern "C" fn helper_string_packsize(context: usize, out: *mut i64, tag: i64, bits: i64) -> i64 {
    // SAFETY: see `variadic!`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let mut words = [0_i64; 1];
    if !text::integers(
        runtime,
        (bank, roots),
        DirectNativeKind::StringPacksize,
        &[TraceInputSlot::from_words(tag, bits)],
        &mut words,
    ) {
        return 1;
    }
    write_words(out, &words);
    0
}

/// `utf8.codepoint` that gave `N` code points.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_utf8_codepoint<const N: usize>(
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
    let args = &all[..usize::try_from(count).unwrap_or(0).min(all.len())];
    // SAFETY: see `variadic!`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    let mut words = [0_i64; MAX_UTF8_CODEPOINTS];
    let mut codes = [0_u32; MAX_UTF8_CODEPOINTS];
    match text::codepoints_fast(runtime, (bank, roots), args, &mut codes) {
        Some(count) if count == N => {
            for (word, code) in words.iter_mut().zip(codes) {
                *word = i64::from(code);
            }
        }
        Some(_) => return 1,
        None => {
            if !text::integers(
                runtime,
                (bank, roots),
                DirectNativeKind::Utf8Codepoint,
                args,
                &mut words[..N],
            ) {
                return 1;
            }
        }
    }
    write_words(out, &words[..N]);
    0
}

variadic!(
    helper_utf8_len_fail,
    [t0 b0, t1 b1, t2 b2, t3 b3],
    |tables, args, out| {
        let (runtime, bank, roots, _, _) = tables.library();
        let mut words = [0_i64; 1];
        let answered = match text::len_fast(runtime, (bank, roots), args) {
            Some(Len::Invalid(position)) => match i64::try_from(position) {
                Ok(position) => {
                    words[0] = position;
                    true
                }
                Err(_) => false,
            },
            Some(Len::Count(_)) => false,
            None => text::failure(
                runtime,
                (bank, roots),
                DirectNativeKind::Utf8Len,
                args,
                Failure::NilAndPosition,
                &mut words,
            ),
        };
        if answered {
            write_words(out, &words);
        }
        answered
    }
);

variadic!(
    helper_utf8_offset_nil,
    [t0 b0, t1 b1, t2 b2],
    |tables, args, out| {
        let _ = out;
        let (runtime, bank, roots, _, _) = tables.library();
        match text::offset_fast((bank, roots), args) {
            Some(Offset::Absent) => true,
            Some(Offset::Found { .. }) => false,
            None => text::failure(
                runtime,
                (bank, roots),
                DirectNativeKind::Utf8Offset,
                args,
                Failure::Nil,
                &mut [],
            ),
        }
    }
);

/// The native of the `utf8.codes` step `mode` names: 0 the strict one, 1 the
/// lax one, 2 Lua 5.3's.
fn step_kind(mode: i64) -> Option<DirectNativeKind> {
    match mode {
        0 => Some(DirectNativeKind::Utf8Next),
        1 => Some(DirectNativeKind::Utf8NextLax),
        2 => Some(DirectNativeKind::Utf8Next53),
        _ => None,
    }
}

/// A step of a `utf8.codes` loop that gives the next character: its position
/// and its code point.
#[allow(unsafe_code)]
extern "C" fn helper_utf8_next(
    context: usize,
    out: *mut i64,
    mode: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
) -> i64 {
    let Some(kind) = step_kind(mode) else {
        return 1;
    };
    let args = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
    ];
    // SAFETY: see `variadic!`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    if let Some(step) = text::codes_next((bank, roots), mode, &args) {
        // A character is the answer, and the end or an invalid sequence is
        // the other helper's or the interpreter's.
        let CodesStep::Character { position, code } = step else {
            return 1;
        };
        let Ok(position) = i64::try_from(position) else {
            return 1;
        };
        write_words(out, &[position, i64::from(code)]);
        return 0;
    }
    let mut words = [0_i64; 2];
    if !text::integers(runtime, (bank, roots), kind, &args, &mut words) {
        return 1;
    }
    write_words(out, &words);
    0
}

/// The step of a `utf8.codes` loop that ends the iteration.
#[allow(unsafe_code)]
extern "C" fn helper_utf8_next_end(
    context: usize,
    _out: *mut i64,
    mode: i64,
    t0: i64,
    b0: i64,
    t1: i64,
    b1: i64,
) -> i64 {
    let Some(kind) = step_kind(mode) else {
        return 1;
    };
    let args = [
        TraceInputSlot::from_words(t0, b0),
        TraceInputSlot::from_words(t1, b1),
    ];
    // SAFETY: see `variadic!`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, _, _) = tables.library();
    if let Some(step) = text::codes_next((bank, roots), mode, &args) {
        return i64::from(step != CodesStep::End);
    }
    i64::from(!text::failure(
        runtime,
        (bank, roots),
        kind,
        &args,
        Failure::Nothing,
        &mut [],
    ))
}

/// The check before `utf8.codes (s)`.
#[allow(unsafe_code)]
extern "C" fn helper_utf8_codes_check(context: usize, _out: *mut i64, tag: i64, bits: i64) -> i64 {
    // SAFETY: see `variadic!`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (_, bank, roots, _, _) = tables.library();
    i64::from(!text::codes_check(
        (bank, roots),
        &[TraceInputSlot::from_words(tag, bits)],
    ))
}

/// `string.unpack` that gave `VALUES` values of the kinds `KINDS` says.
#[allow(unsafe_code, clippy::too_many_arguments)]
extern "C" fn helper_string_unpack<const VALUES: usize, const KINDS: usize>(
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
    let args = &all[..usize::try_from(count).unwrap_or(0).min(all.len())];
    // SAFETY: see `variadic!`.
    let tables = unsafe { &mut *(context as *mut RunTables<'static>) };
    let (runtime, bank, roots, state, lengths) = tables.library();
    let mut words = [0_i64; MAX_UNPACK_VALUES + 1];
    if !text::unpack(
        runtime,
        bank,
        roots,
        (state, lengths),
        args,
        (VALUES, KINDS),
        &mut words[..=VALUES],
    ) {
        return 1;
    }
    write_words(out, &words[..=VALUES]);
    0
}

/// Installs the addresses of the helpers of this module.
pub(super) fn install() {
    install_helper(HelperId::UTF8_CHAR, helper_utf8_char as *const () as usize);
    install_helper(HelperId::UTF8_LEN, helper_utf8_len as *const () as usize);
    install_helper(
        HelperId::UTF8_LEN_FAIL,
        helper_utf8_len_fail as *const () as usize,
    );
    install_helper(
        HelperId::UTF8_OFFSET,
        helper_utf8_offset as *const () as usize,
    );
    install_helper(
        HelperId::UTF8_OFFSET_END,
        helper_utf8_offset_end as *const () as usize,
    );
    install_helper(
        HelperId::UTF8_OFFSET_NIL,
        helper_utf8_offset_nil as *const () as usize,
    );
    install_helper(HelperId::UTF8_NEXT, helper_utf8_next as *const () as usize);
    install_helper(
        HelperId::UTF8_NEXT_END,
        helper_utf8_next_end as *const () as usize,
    );
    install_helper(
        HelperId::UTF8_CODES_CHECK,
        helper_utf8_codes_check as *const () as usize,
    );
    install_helper(
        HelperId::STRING_PACK,
        helper_string_pack as *const () as usize,
    );
    install_helper(
        HelperId::STRING_PACKSIZE,
        helper_string_packsize as *const () as usize,
    );
    let codepoints = [
        helper_utf8_codepoint::<0> as *const () as usize,
        helper_utf8_codepoint::<1> as *const () as usize,
        helper_utf8_codepoint::<2> as *const () as usize,
        helper_utf8_codepoint::<3> as *const () as usize,
        helper_utf8_codepoint::<4> as *const () as usize,
    ];
    for (count, function) in codepoints.into_iter().enumerate() {
        install_helper(HelperId::utf8_codepoint(count), function);
    }
    let unpacks: [((usize, usize), usize); 40] = [
        ((0, 0), helper_string_unpack::<0, 0> as *const () as usize),
        ((1, 0), helper_string_unpack::<1, 0> as *const () as usize),
        ((1, 1), helper_string_unpack::<1, 1> as *const () as usize),
        ((1, 2), helper_string_unpack::<1, 2> as *const () as usize),
        ((2, 0), helper_string_unpack::<2, 0> as *const () as usize),
        ((2, 1), helper_string_unpack::<2, 1> as *const () as usize),
        ((2, 2), helper_string_unpack::<2, 2> as *const () as usize),
        ((2, 3), helper_string_unpack::<2, 3> as *const () as usize),
        ((2, 4), helper_string_unpack::<2, 4> as *const () as usize),
        ((2, 5), helper_string_unpack::<2, 5> as *const () as usize),
        ((2, 6), helper_string_unpack::<2, 6> as *const () as usize),
        ((2, 7), helper_string_unpack::<2, 7> as *const () as usize),
        ((2, 8), helper_string_unpack::<2, 8> as *const () as usize),
        ((3, 0), helper_string_unpack::<3, 0> as *const () as usize),
        ((3, 1), helper_string_unpack::<3, 1> as *const () as usize),
        ((3, 2), helper_string_unpack::<3, 2> as *const () as usize),
        ((3, 3), helper_string_unpack::<3, 3> as *const () as usize),
        ((3, 4), helper_string_unpack::<3, 4> as *const () as usize),
        ((3, 5), helper_string_unpack::<3, 5> as *const () as usize),
        ((3, 6), helper_string_unpack::<3, 6> as *const () as usize),
        ((3, 7), helper_string_unpack::<3, 7> as *const () as usize),
        ((3, 8), helper_string_unpack::<3, 8> as *const () as usize),
        ((3, 9), helper_string_unpack::<3, 9> as *const () as usize),
        ((3, 10), helper_string_unpack::<3, 10> as *const () as usize),
        ((3, 11), helper_string_unpack::<3, 11> as *const () as usize),
        ((3, 12), helper_string_unpack::<3, 12> as *const () as usize),
        ((3, 13), helper_string_unpack::<3, 13> as *const () as usize),
        ((3, 14), helper_string_unpack::<3, 14> as *const () as usize),
        ((3, 15), helper_string_unpack::<3, 15> as *const () as usize),
        ((3, 16), helper_string_unpack::<3, 16> as *const () as usize),
        ((3, 17), helper_string_unpack::<3, 17> as *const () as usize),
        ((3, 18), helper_string_unpack::<3, 18> as *const () as usize),
        ((3, 19), helper_string_unpack::<3, 19> as *const () as usize),
        ((3, 20), helper_string_unpack::<3, 20> as *const () as usize),
        ((3, 21), helper_string_unpack::<3, 21> as *const () as usize),
        ((3, 22), helper_string_unpack::<3, 22> as *const () as usize),
        ((3, 23), helper_string_unpack::<3, 23> as *const () as usize),
        ((3, 24), helper_string_unpack::<3, 24> as *const () as usize),
        ((3, 25), helper_string_unpack::<3, 25> as *const () as usize),
        ((3, 26), helper_string_unpack::<3, 26> as *const () as usize),
    ];
    for ((values, kinds), function) in unpacks {
        install_helper(HelperId::string_unpack(values, kinds), function);
    }
}
