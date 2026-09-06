//! `utf8.*` and `string.pack`, `string.unpack` and `string.packsize` in a
//! trace: the loop compiles, the trace calls the helper of the call, and the
//! results (and the errors) are the interpreter's. Lua 5.3 and later install
//! these functions. The other profiles have none, so the same scripts fail
//! in them the way the interpreter does.

use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

/// Iterations of the loops of [`looped`].
const ITERATIONS: u64 = 20000;

/// Whether the profile installs `utf8` and `string.pack`.
fn installs(profile: CompatibilityProfile) -> bool {
    matches!(
        profile,
        CompatibilityProfile::Lua53 | CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
    )
}

fn outcome(
    profile: CompatibilityProfile,
    mode: ExecutionMode,
    source: &str,
) -> (Result<Vec<Value>, String>, Lua) {
    let mut lua = Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap();
    lua.vm_mut().use_synchronous_jit_compilation();
    lua.vm_mut().ignore_remaining_work();
    let results = lua.exec(source).map_err(|error| format!("{error:?}"));
    (results, lua)
}

/// Runs `source` traced and interpreted in every profile and checks that the
/// outcomes agree. In a profile that installs the functions, the trace must
/// also have run natively for at least `floor` iterations, calling each of
/// `helpers`.
fn check(source: &str, helpers: &[&str], floor: u64) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, lua) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
        if !installs(profile) {
            continue;
        }
        let stats = lua.vm().jit_statistics().trace;
        assert!(stats.traces_compiled > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.native_entries > 0, "{profile:?} {source} {stats:?}");
        assert!(stats.iterations >= floor, "{profile:?} {source} {stats:?}");
        for helper in helpers {
            // A name with a bar stands for any one of the helpers it lists.
            let calls = lua.vm().trace_helper_calls();
            assert!(
                helper.split('|').any(|one| calls.contains(&one)),
                "{profile:?} {source}: wanted {helper}, got {:?}",
                lua.vm().trace_helper_calls()
            );
        }
    }
}

/// [`check`] for a loop that runs natively almost all of its iterations.
fn running(source: &str, helpers: &[&str]) {
    check(source, helpers, ITERATIONS * 9 / 10);
}

/// The outcomes agree in every profile, whether or not a trace runs (a call
/// the recorder leaves to the interpreter).
fn agrees(source: &str) {
    for profile in PROFILES {
        let (expected, _) = outcome(profile, ExecutionMode::Interpreter, source);
        let (actual, _) = outcome(profile, ExecutionMode::Tracing, source);
        assert_eq!(actual, expected, "{profile:?} {source}");
    }
}

/// A loop over `body`, which sums into `n`, after `setup`.
fn looped(setup: &str, body: &str) -> String {
    format!("local n = 0 {setup} for i = 1, {ITERATIONS} do {body} end return n")
}

const TEXT: &str = "local s = 'a\\u{E9}\\u{20AC}x\\u{1F600}z'";

/// A subject of 1500 bytes made of the characters of [`TEXT`], and one of
/// ASCII, for the loops that walk a string.
const LONG: &str =
    "local s = ('a\\u{E9}\\u{20AC}x\\u{1F600}z'):rep(120) local a = ('abcdefgh'):rep(200)";

#[test]
fn char_with_no_one_and_several_arguments() {
    let cases = [
        "n = n + #utf8.char()",
        "n = n + #utf8.char(i % 100 + 20)",
        "n = n + #utf8.char(72, 228)",
        "n = n + #utf8.char(72, 228, 8364, 0x1F600)",
        "n = n + #utf8.char(65, 66, 67, 68, 69)",
        "n = n + #utf8.char(0x7F) + #utf8.char(0x80) + #utf8.char(0x7FF) + #utf8.char(0x800)",
        "n = n + #utf8.char(0xFFFF) + #utf8.char(0x10000) + #utf8.char(0x10FFFF)",
        "n = n + #utf8.char(0xD800)",
        "n = n + #utf8.char(65.0, '66')",
    ];
    for body in cases {
        running(&looped("", body), &["utf8.char"]);
    }
}

#[test]
fn char_past_the_limit_of_the_profile_exits() {
    // Lua 5.3 stops at 0x10FFFF, the later profiles at 0x7FFFFFFF.
    for limit in ["0x10FFFF", "0x110000", "0x7FFFFFFF", "0x80000000", "-1"] {
        let source = looped(
            "",
            &format!("n = n + #utf8.char(i < 19000 and 65 or {limit})"),
        );
        check(&source, &["utf8.char"], 18000);
    }
}

#[test]
fn codepoint_over_ranges_and_negative_indices() {
    let cases = [
        "n = n + utf8.codepoint(s)",
        "n = n + utf8.codepoint(s, 2)",
        "n = n + utf8.codepoint(s, -1)",
        "n = n + utf8.codepoint(s, -5)",
        "local a, b = utf8.codepoint(s, 1, 3) n = n + a + b",
        "local a, b, c = utf8.codepoint(s, 1, 6) n = n + a + b + c",
        "local a, b, c, d = utf8.codepoint(s, 1, 7) n = n + a + b + c + d",
        "local a, b = utf8.codepoint(s, -6, -1) n = n + a + b",
        // An empty range gives no value.
        "n = n + select('#', utf8.codepoint(s, 3, 2))",
        "n = n + select('#', utf8.codepoint(s, 2, 2))",
        "n = n + (utf8.codepoint(s, 4, 3) or 5)",
        "n = n + utf8.codepoint('\\u{10FFFF}')",
    ];
    for body in cases {
        running(&looped(TEXT, body), &["utf8.codepoint"]);
    }
}

#[test]
fn codepoint_with_a_changing_count_and_out_of_range_indices() {
    // The count changes with the iteration: side traces take the others.
    check(
        &looped(
            TEXT,
            "local a, b, c = utf8.codepoint(s, 1, i % 3 + 1) n = n + a + (b or 0) + (c or 0)",
        ),
        &["utf8.codepoint"],
        15000,
    );
    // More code points than a helper holds are the interpreter's.
    agrees(&looped(
        TEXT,
        "local t = {utf8.codepoint(s, 1, -1)} n = n + #t",
    ));
    agrees(&looped(
        TEXT,
        "n = n + select('#', utf8.codepoint(s, 1, -1))",
    ));
    agrees(&looped(
        TEXT,
        "local a, b, c, d, e = utf8.codepoint(s, 1, -1) n = n + a + b + c + d + e",
    ));
    // Errors: a start before the string, an end past it, a bad sequence.
    for call in [
        "utf8.codepoint(s, i < 19000 and 1 or 0)",
        "utf8.codepoint(s, 1, i < 19000 and 1 or 99)",
        "utf8.codepoint(i < 19000 and s or bad)",
        "utf8.codepoint(i < 19000 and s or cut)",
        "utf8.codepoint(s, i < 19000 and 1 or 3)",
    ] {
        let setup = format!("{TEXT} local bad, cut = '\\xff', '\\xe2\\x82'");
        check(&looped(&setup, &format!("n = n + {call}")), &[], 15000);
    }
}

#[test]
fn len_on_valid_and_invalid_strings() {
    // The arguments after the string, and the helper that answers.
    let cases = [
        ("s", "utf8.len"),
        ("s, 2", "utf8.len|utf8.len_fail"),
        ("s, -4", "utf8.len|utf8.len_fail"),
        ("s, 1, 3", "utf8.len|utf8.len_fail"),
        ("s, 3, 2", "utf8.len"),
        ("''", "utf8.len"),
        ("s, #s + 1", "utf8.len"),
        ("s, 1, 12", "utf8.len"),
        ("s, 4, -1", "utf8.len"),
        // The failure: nil and the position of the invalid byte.
        ("'ab\\xffcd'", "utf8.len_fail"),
        ("s .. '\\xe2\\x82'", "utf8.len_fail"),
        // Lua 5.3 takes surrogates and code points past 0x10FFFF.
        ("'\\xed\\xa0\\x80'", "utf8.len|utf8.len_fail"),
        ("'\\xf4\\x90\\x80\\x80'", "utf8.len|utf8.len_fail"),
        ("'\\xc0\\x80'", "utf8.len_fail"),
        ("'\\x80'", "utf8.len_fail"),
        // The lax flag of Lua 5.4 and later takes surrogates and large
        // code points.
        ("'\\xed\\xa0\\x80', 1, -1, true", "utf8.len|utf8.len_fail"),
        (
            "'\\xf4\\x90\\x80\\x80', 1, -1, true",
            "utf8.len|utf8.len_fail",
        ),
        (
            "'\\xfd\\xbf\\xbf\\xbf\\xbf\\xbf', 1, -1, true",
            "utf8.len|utf8.len_fail",
        ),
        ("s, 1, -1, false", "utf8.len"),
    ];
    for (arguments, helper) in cases {
        let body = format!("local a, b = utf8.len({arguments}) n = n + (a or -1) + (b or 0)");
        running(&looped(TEXT, &body), &[helper]);
    }
}

#[test]
fn len_that_flips_between_a_count_and_a_failure() {
    for modulus in [2, 50] {
        check(
            &looped(
                TEXT,
                &format!(
                    "local t = (i % {modulus} == 0) and s or 'a\\xffc' \
                     local a, b = utf8.len(t) n = n + (a or -1) + (b or 0)"
                ),
            ),
            &["utf8.len"],
            15000,
        );
    }
}

#[test]
fn offset_forward_backward_zero_and_past_the_end() {
    let cases = [
        "n = n + utf8.offset(s, 1)",
        "n = n + utf8.offset(s, 3)",
        "n = n + utf8.offset(s, 6)",
        "n = n + (utf8.offset(s, 7) or 100)",
        "n = n + (utf8.offset(s, 8) or 100)",
        "n = n + utf8.offset(s, -1)",
        "n = n + utf8.offset(s, -3)",
        "n = n + (utf8.offset(s, -7) or 100)",
        "n = n + utf8.offset(s, 0)",
        "n = n + utf8.offset(s, 0, 3)",
        "n = n + utf8.offset(s, 0, 4)",
        "n = n + utf8.offset(s, 2, 2)",
        "n = n + utf8.offset(s, -2, #s + 1)",
        "n = n + utf8.offset(s, 1, #s + 1)",
        // Lua 5.5 gives the end of the character too.
        "local a, b = utf8.offset(s, 3) n = n + a + (b or 0)",
        "local a, b = utf8.offset(s, 0, 3) n = n + a + (b or 0)",
    ];
    for body in cases {
        running(
            &looped(TEXT, body),
            &["utf8.offset|utf8.offset_end|utf8.offset_nil"],
        );
    }
    running(
        &looped(TEXT, "n = n + (utf8.offset(s, 9) or 100)"),
        &["utf8.offset_nil"],
    );
    running(
        &looped(TEXT, "n = n + (utf8.offset(s, -9) or 100)"),
        &["utf8.offset_nil"],
    );
    running(
        &looped(TEXT, "n = n + utf8.offset(s, 4)"),
        &["utf8.offset|utf8.offset_end"],
    );
    // A start on a continuation byte raises.
    check(
        &looped(TEXT, "n = n + utf8.offset(s, 2, i < 19000 and 1 or 3)"),
        &[],
        15000,
    );
    // A position that is found, then not.
    check(
        &looped(TEXT, "n = n + (utf8.offset(s, i % 40 + 1) or 100)"),
        &[],
        10000,
    );
}

#[test]
fn codes_over_ascii_multibyte_and_lax_strings() {
    let cases = [
        "for p, c in utf8.codes(a) do n = n + p + c end",
        "for p, c in utf8.codes(s) do n = n + p + c end",
        "for p in utf8.codes(s) do n = n + p end",
        "for p, c in utf8.codes('') do n = n + p + c end n = n + 1",
        "for p, c in utf8.codes(s, true) do n = n + p + c end",
        "for p, c in utf8.codes(s, false) do n = n + p + c end",
        "for p, c in utf8.codes(s, nil) do n = n + p + c end",
        "for p, c in utf8.codes('\\u{7FFFFFFF}', true) do n = n + p + c end",
    ];
    for body in cases {
        let source = format!("local n = 0 {LONG} for r = 1, 40 do {body} end return n");
        // The empty string and the one character loop too little to compile.
        if body.contains("utf8.codes('") {
            agrees(&source);
        } else {
            check(&source, &["utf8.next"], 20000);
        }
    }
}

#[test]
fn codes_compiles_a_long_loop_and_calls_the_step() {
    let source = "local n = 0 local s = ('a\\u{E9}\\u{20AC}x\\u{1F600}z'):rep(4000) \
                  for r = 1, 5 do for p, c in utf8.codes(s) do n = n + p + c end end return n";
    check(source, &["utf8.next"], 90000);
    let ascii = "local n = 0 local s = ('abcdefgh'):rep(5000) \
                 for r = 1, 3 do for p, c in utf8.codes(s) do n = n + c end end return n";
    check(ascii, &["utf8.next"], 100_000);
    let lax = "local n = 0 local s = ('a\\u{E9}\\u{20AC}x'):rep(5000) \
               for r = 1, 3 do for p, c in utf8.codes(s, true) do n = n + c end end return n";
    check(lax, &["utf8.next"], 50000);
}

#[test]
fn codes_nested_in_an_outer_loop() {
    let source = "local n = 0 local s = 'a\\u{E9}\\u{20AC}x\\u{1F600}z' \
                  for r = 1, 3000 do for p, c in utf8.codes(s) do n = n + p + c end \
                  n = n + utf8.len(s) end return n";
    check(source, &["utf8.next", "utf8.next_end"], 3000);
    // A different string each time.
    let source = "local n = 0 local words = {'abc', '\\u{E9}t\\u{E9}', '\\u{20AC}', 'x'} \
                  for r = 1, 4000 do for p, c in utf8.codes(words[r % 4 + 1]) do n = n + c end end return n";
    agrees(source);
    // The iterator held in a variable, and a loop over a string made in it.
    let source = "local n = 0 local s = ('h\\u{E9}llo'):rep(500) \
                  for r = 1, 30 do local f, st, c = utf8.codes(s) \
                  while true do local p, cp = f(st, c) if not p then break end c = p n = n + cp end end return n";
    agrees(source);
    let source = "local n = 0 for r = 1, 400 do local s = ('h\\u{E9}'):rep(r % 30 + 1) \
                  for p, c in utf8.codes(s) do n = n + c end end return n";
    agrees(source);
}

#[test]
fn codes_over_invalid_input_exits_and_raises_the_interpreters_error() {
    for bad in [
        "ab\\xffcd",
        "\\x80abc",
        "ab\\xe2\\x82",
        "ab\\xc0\\x80",
        "a\\xed\\xa0\\x80",
        "ab\\xf4\\x90\\x80\\x80",
        "abc\\x80",
    ] {
        for flag in ["", ", true"] {
            // The bad string comes after good ones, and in the middle of a
            // long run of iterations.
            let source = format!(
                "local n = 0 local good = ('abc\\u{{E9}}'):rep(400) local bad = '{bad}' \
                 for r = 1, 40 do local s = r < 40 and good or bad \
                 for p, c in utf8.codes(s{flag}) do n = n + c end end return n"
            );
            // 39 good passes of 1600 characters, then the error.
            check(&source, &["utf8.next"], 20000);
            let long = format!(
                "local n = 0 local s = ('abc\\u{{E9}}'):rep(3000) .. '{bad}' \
                 for p, c in utf8.codes(s{flag}) do n = n + c end return n"
            );
            // 12000 good characters, then the error (or, in a lax loop, the
            // end of a string the lax decoder accepts).
            check(&long, &["utf8.next"], 7000);
        }
    }
}

#[test]
fn codes_that_stops_early_and_with_the_step_in_a_variable() {
    let source = "local n = 0 local s = ('abc\\u{E9}'):rep(800) \
                  for r = 1, 40 do for p, c in utf8.codes(s) do n = n + c if p > 500 then break end end end return n";
    agrees(source);
    check(source, &[], 1000);
}

#[test]
fn pack_and_unpack_round_trip_each_format_class() {
    // Format, the value and the sum that reads the result back.
    let cases = [
        ("<i1", "i % 100"),
        (">i2", "i % 30000"),
        ("<I3", "i * 7"),
        ("=i4", "-i"),
        (">i8", "i * 100000"),
        ("<j", "-i * 3"),
        ("<I2", "i % 60000"),
        ("<B", "i % 256"),
        ("<b", "i % 100 - 50"),
        ("<h", "i % 100 - 50"),
        ("<H", "i % 100"),
        ("<l", "i"),
        ("<L", "i"),
        ("<T", "i"),
        ("<i16", "i"),
        ("<f", "i / 4"),
        (">d", "i / 8"),
        ("<n", "i * 0.5"),
    ];
    for (format, value) in cases {
        let body = format!(
            "local x = string.pack('{format}', {value}) n = n + #x \
             local a, p = string.unpack('{format}', x) n = n + a + p"
        );
        running(&looped("", &body), &["string.pack", "string.unpack"]);
    }
    let strings = [
        ("<s1", "'abc' .. i % 10"),
        ("<s2", "'abc' .. i % 10"),
        (">s4", "'abc' .. i % 10"),
        ("<s", "'abc' .. i % 10"),
        ("z", "'abc' .. i % 10"),
        ("c6", "'abc' .. i % 10"),
    ];
    for (format, value) in strings {
        let body = format!(
            "local x = string.pack('{format}', {value}) n = n + #x \
             local a, p = string.unpack('{format}', x) n = n + #a + p"
        );
        running(&looped("", &body), &["string.pack", "string.unpack"]);
    }
}

#[test]
fn pack_and_unpack_several_values_alignment_and_padding() {
    let cases = [
        // Several values: up to three, then the next position.
        (
            "'<i4I2z'",
            "i, i % 60000, 'ab'",
            "local a, b, c, p = string.unpack('<i4I2z', x) n = n + a + b + #c + p",
        ),
        (
            "'<i2i2i2'",
            "i % 100, 2, 3",
            "local a, b, c, p = string.unpack('<i2i2i2', x) n = n + a + b + c + p",
        ),
        (
            "'<i1 d s1'",
            "i % 100, 0.5, 'q'",
            "local a, b, c, p = string.unpack('<i1 d s1', x) n = n + a + b + #c + p",
        ),
        // Endianness switches inside one format.
        (
            "'<i2>i2=i2'",
            "i % 100, i % 50, 7",
            "local a, b, c, p = string.unpack('<i2>i2=i2', x) n = n + a + b + c + p",
        ),
        // Alignment.
        (
            "'<!4 i1 i4'",
            "1, i",
            "local a, b, p = string.unpack('<!4 i1 i4', x) n = n + a + b + p",
        ),
        (
            "'<!8 i1 d i2'",
            "1, 2.5, 3",
            "local a, b, c, p = string.unpack('<!8 i1 d i2', x) n = n + a + b + c + p",
        ),
        (
            "'<!4 i1 Xi4 i1'",
            "1, 2",
            "local a, b, p = string.unpack('<!4 i1 Xi4 i1', x) n = n + a + b + p",
        ),
        // Padding.
        (
            "'<i1 x x i2'",
            "i % 100, 3",
            "local a, b, p = string.unpack('<i1 x x i2', x) n = n + a + b + p",
        ),
        // An initial position, positive and negative.
        (
            "'<i2i2'",
            "i % 100, 3",
            "local a, p = string.unpack('<i2', x, 3) n = n + a + p",
        ),
        (
            "'<i2i2'",
            "i % 100, 3",
            "local a, p = string.unpack('<i2', x, -2) n = n + a + p",
        ),
        (
            "'<i2i2'",
            "i % 100, 3",
            "local a, p = string.unpack('<i2', x, 1) n = n + a + p",
        ),
        // Values that are not all of one kind.
        (
            "'<s1 s1'",
            "'a' .. i % 7, 'bc'",
            "local a, b, p = string.unpack('<s1 s1', x) n = n + #a + #b + p",
        ),
        (
            "'<f f'",
            "1.5, i",
            "local a, b, p = string.unpack('<f f', x) n = n + a + b + p",
        ),
    ];
    for (format, values, read) in cases {
        let body = format!("local x = string.pack({format}, {values}) n = n + #x {read}");
        running(&looped("", &body), &["string.pack", "string.unpack"]);
    }
}

#[test]
fn more_values_than_a_helper_holds_are_the_interpreters() {
    let source = looped(
        "",
        "local a, b, c, d, e, p = string.unpack('<i1i1i1i1i1', '\\1\\2\\3\\4\\5') n = n + a + b + c + d + e + p",
    );
    agrees(&source);
    let source = looped(
        "",
        "local x = string.pack('<i1i1i1i1i1i1', 1, 2, 3, 4, 5, 6) n = n + #x",
    );
    agrees(&source);
    // A variable format.
    let source = looped(
        "local formats = {'<i2', '>i2', '<I1'}",
        "local f = formats[i % 3 + 1] local x = string.pack(f, i % 100) n = n + #x + (string.unpack(f, x))",
    );
    check(&source, &["string.pack", "string.unpack"], 15000);
}

#[test]
fn packsize_of_fixed_formats() {
    for format in [
        "<i4i2",
        "i1",
        "<!4 i1 i4",
        "!8 i1 d",
        "c10",
        "<j",
        "x",
        "i16",
        "<!2 i1 Xi8",
    ] {
        running(
            &looped("", &format!("n = n + string.packsize('{format}')")),
            &["string.packsize"],
        );
    }
    // A format without a fixed size raises.
    check(
        &looped(
            "local good, bad = 'i4', 'z'",
            "n = n + string.packsize(i < 19000 and good or bad)",
        ),
        &["string.packsize"],
        15000,
    );
}

#[test]
fn a_changing_value_kind_takes_a_side_trace() {
    // The value unpacked is an integer, then a float.
    let source = looped(
        "local ints, floats = '<i4', '<f'",
        "local f = (i % 2 == 0) and ints or floats local x = string.pack(f, i) \
         local v, p = string.unpack(f, x) n = n + v + p",
    );
    check(&source, &["string.unpack"], 10000);
    // The value is a string, then an integer.
    let source = looped(
        "local ints, strings = '<i1', '<s1'",
        "local f = (i % 4 == 0) and ints or strings local x = string.pack(f, (i % 4 == 0) and 7 or 'a') \
         local v, p = string.unpack(f, x) n = n + (tonumber(v) or 1) + p",
    );
    agrees(&source);
}

#[test]
fn pack_overflow_unpack_past_the_end_and_invalid_formats_exit() {
    // `good` is used for most of the iterations, `bad` for the last ones.
    let cases = [
        // Integer overflow in pack.
        ("string.pack('i1', V)", "5", "200"),
        ("string.pack('I2', V)", "5", "-1"),
        ("string.pack('i4', V)", "5", "2^40"),
        ("string.pack('i4', V)", "5", "0.5"),
        // Extra and missing arguments.
        ("string.pack(V, 1)", "'i4'", "'i4i4'"),
        ("string.pack(V, 1, 2)", "'i4i4'", "'i4'"),
        // Too long a string for s1, zeros in z, a string too long for c3.
        ("string.pack('s1', V)", "'ab'", "('x'):rep(300)"),
        ("string.pack('z', V)", "'ab'", "'a\\0b'"),
        ("string.pack('c3', V)", "'ab'", "'abcd'"),
        // Unpack past the end of the data, of a string, of a position.
        ("string.unpack('<i4', V)", "'\\1\\2\\3\\4'", "'\\1\\2'"),
        ("string.unpack('z', V)", "'ab\\0'", "'ab'"),
        ("string.unpack('s1', V)", "'\\2ab'", "'\\9ab'"),
        ("string.unpack('<i2', '\\1\\2\\3\\4', V)", "1", "9"),
        ("string.unpack('<i2', '\\1\\2\\3\\4', V)", "1", "-9"),
        // Lua 5.3 raises for these, the later profiles clamp to the string.
        ("string.unpack('<i2', '\\1\\2\\3\\4', V)", "1", "0"),
        ("string.unpack('<i2', '\\1\\2\\3\\4', V)", "1", "-100"),
        // Invalid formats.
        ("string.pack(V, 1)", "'i1'", "'q'"),
        ("string.pack(V, 1)", "'i1'", "'i17'"),
        ("string.pack(V, 1)", "'i1'", "'i0'"),
        ("string.pack(V, 1)", "'i1'", "'!3'"),
        ("string.packsize(V)", "'i1'", "'c'"),
        ("string.unpack(V, '\\1')", "'<i1'", "'<Xz'"),
        // Wrong argument types.
        ("string.pack('i1', V)", "1", "{}"),
        ("string.pack('i1', V)", "1", "'z'"),
        ("string.pack(V, 1)", "'i1'", "{}"),
        ("string.unpack('<i1', V)", "'\\1'", "{}"),
        ("string.packsize(V)", "'i1'", "nil"),
        ("utf8.len(V)", "'abc'", "{}"),
        ("utf8.len(V)", "'abc'", "nil"),
        ("utf8.char(V)", "65", "{}"),
        ("utf8.char(V)", "65", "'x'"),
        ("utf8.codepoint(V)", "'abc'", "{}"),
        ("utf8.offset(V, 1)", "'abc'", "{}"),
        ("utf8.offset('abc', V)", "1", "{}"),
        ("utf8.len('abc', V)", "1", "'x'"),
        ("utf8.len('abc', V)", "1", "9"),
    ];
    for (call, good, bad) in cases {
        let call = call.replace('V', "(i < 19000 and good or bad)");
        let body = format!("local r = {call} n = n + (type(r) == 'number' and r or #r)");
        let setup = format!("local good, bad = {good}, {bad}");
        check(&looped(&setup, &body), &[], 15000);
    }
}

#[test]
fn a_replaced_function_is_the_interpreters() {
    let cases = [
        "if i == 15000 then utf8.len = function() return 7 end end n = n + utf8.len('abc')",
        "if i == 15000 then utf8.char = function() return 'qq' end end n = n + #utf8.char(65, 66)",
        "if i == 15000 then utf8.codepoint = function() return 5 end end n = n + utf8.codepoint('abc')",
        "if i == 15000 then utf8.offset = function() return 5 end end n = n + utf8.offset('abc', 2)",
        "if i == 15000 then string.pack = function() return 'xx' end end n = n + #string.pack('i2', i)",
        "if i == 15000 then string.unpack = function() return 5, 6 end end n = n + (string.unpack('i1', 'a'))",
        "if i == 15000 then string.packsize = function() return 9 end end n = n + string.packsize('i4')",
        "if i == 15000 then utf8.codes = function() return function() end end end for p, c in utf8.codes('ab') do n = n + c end n = n + 1",
    ];
    for body in cases {
        check(&looped("", body), &[], 14000);
    }
}

#[test]
fn utf8_charpattern_is_a_constant_in_a_trace() {
    // The field holds a string the entry checks, and the trace reads it as a
    // constant. The bodies carry some arithmetic so the loop is long enough
    // to compile.
    let work = "(i % 7) * 3 + (i // 5) % 11";
    running(
        &looped("", &format!("n = n + #utf8.charpattern + {work}")),
        &[],
    );
    running(
        &looped(
            "",
            &format!("local p = utf8.charpattern n = n + #p + {work}"),
        ),
        &[],
    );
    running(
        &looped(
            "",
            &format!("n = n + {work} + (string.find('a\\u{{E9}}', utf8.charpattern, 2) or 0)"),
        ),
        &[],
    );
    // An outer loop over a gmatch loop on the pattern.
    check(
        &looped(
            TEXT,
            &format!(
                "local k = 0 for c in s:gmatch(utf8.charpattern) do k = k + 1 end n = n + k + {work}"
            ),
        ),
        &[],
        ITERATIONS / 2,
    );
    // A replaced field is the interpreter's.
    check(
        &looped(
            "",
            &format!(
                "if i == 15000 then utf8.charpattern = 'x' end n = n + #utf8.charpattern + {work}"
            ),
        ),
        &[],
        14000,
    );
}

#[test]
fn text_loops_that_decode_and_encode() {
    let source = "local n = 0 local out = {} \
                  for i = 1, 20000 do \
                      local s = utf8.char(0x41 + i % 26, 0xE9, 0x20AC) \
                      local len = utf8.len(s) \
                      local a, b, c = utf8.codepoint(s, 1, #s) \
                      n = n + a + b + c + len + utf8.offset(s, 3) \
                  end return n";
    running(
        source,
        &[
            "utf8.char",
            "utf8.len",
            "utf8.codepoint",
            "utf8.offset|utf8.offset_end",
        ],
    );
}

#[test]
fn strings_read_from_a_table_are_arguments_too() {
    let setup = "local words = {'abc', 'h\\u{E9}llo', '\\u{20AC}x', '\\u{1F600}'} \
                 local packed = {} for k = 1, #words do packed[k] = string.pack('<s1i2', words[k], k) end";
    for body in [
        "local w = words[i % 4 + 1] n = n + utf8.len(w) + utf8.codepoint(w, 1) + (utf8.offset(w, 2) or 0)",
        "local w = words[i % 4 + 1] n = n + #utf8.char(utf8.codepoint(w, 1))",
        "local s, k, p = string.unpack('<s1i2', packed[i % 4 + 1]) n = n + #s + k + p",
    ] {
        running(&looped(setup, body), &[]);
    }
    // A short loop over the string of the table is a loop inside a loop.
    agrees(&looped(
        setup,
        "local w = words[i % 4 + 1] for p, c in utf8.codes(w) do n = n + p + c end",
    ));
}

#[test]
fn records_around_the_sizes_a_pack_buffer_keeps_in_place() {
    // Packed records of 21 to 23 bytes are inline strings, the ones of 47 to
    // 50 bytes straddle the buffer `string.pack` fills before it moves to a
    // `Vec`, and a padded one is far past it.
    for size in [0, 1, 21, 22, 23, 46, 47, 48, 49, 50, 64, 200] {
        let body = format!(
            "local x = string.pack('<i4 c{size} i2', i, ('x'):rep({size}), i % 100) n = n + #x \
             local a, b, c, p = string.unpack('<i4 c{size} i2', x) n = n + a + #b + c + p"
        );
        running(&looped("", &body), &["string.pack", "string.unpack"]);
    }
    let body = "local x = string.pack('<!16 i1 i16 i1 d', 1, i, 2, 0.5) n = n + #x \
                local a, b, c, d, p = string.unpack('<!16 i1 i16 i1 d', x) n = n + a + b + c + d + p";
    // Four values and the position are more than a helper holds.
    agrees(&looped("", body));
}

#[test]
fn unpack_with_the_most_values_a_helper_holds_and_one_more() {
    let three = "local a, b, c, p = string.unpack('<i2 i2 s1', x) n = n + a + b + #c + p";
    let four = "local a, b, c, d, p = string.unpack('<i2 i2 s1 d', x) \
                n = n + a + b + #c + d + p";
    let five = "local a, b, c, d, e, p = string.unpack('<i2 i2 s1 d i1', x) \
                n = n + a + b + #c + d + e + p";
    let six = "local t = {string.unpack('<i2 i2 s1 d i1 i1 i1', x)} n = n + #t";
    let cases = [
        ("'<i2 i2 s1'", "i % 100, 7, 'ab'", three, true),
        ("'<i2 i2 s1 d'", "i % 100, 7, 'ab', 0.25", four, false),
        ("'<i2 i2 s1 d i1'", "i % 100, 7, 'ab', 0.25, 5", five, false),
        (
            "'<i2 i2 s1 d i1 i1 i1'",
            "i % 100, 7, 'ab', 0.25, 5, 6, 7",
            six,
            false,
        ),
    ];
    for (format, values, read, traced) in cases {
        let body = format!("local x = string.pack({format}, {values}) n = n + #x {read}");
        if traced {
            running(&looped("", &body), &["string.pack", "string.unpack"]);
        } else {
            agrees(&looped("", &body));
        }
    }
}

#[test]
fn pack_and_unpack_errors_after_the_trace_has_run() {
    let cases = [
        // A number where the format wants a string, and an extra argument.
        "string.pack('<s1', i < 19000 and 'ab' or 5.5)",
        "string.pack('<i2', i < 19000 and 1 or 'x')",
        "string.pack('<i2', i < 19000 and 1 or 70000)",
        "string.pack('<i1', 1, i < 19000 and nil or 2)",
        "string.pack(i < 19000 and '<i1' or '<i1 i1', 1)",
        "string.pack('<z', i < 19000 and 'ab' or 'a\\0b')",
        "string.pack('<c2', i < 19000 and 'ab' or 'abc')",
        // The data runs out, the position is outside, the format is bad.
        "string.unpack('<i4', i < 19000 and '\\1\\2\\3\\4' or '\\1\\2\\3')",
        "string.unpack('<i1', '\\1', i < 19000 and 1 or 3)",
        "string.unpack('<s1', i < 19000 and '\\1a' or '\\9a')",
        "string.unpack(i < 19000 and '<i1' or '<q', '\\1')",
        "string.unpack('<z', i < 19000 and 'ab\\0' or 'ab')",
    ];
    for call in cases {
        agrees(&looped("", &format!("n = n + #tostring({call})")));
    }
}

#[test]
fn utf8_calls_whose_results_are_a_mix_of_counts_and_failures() {
    let cases = [
        "local a, b = utf8.len(i % 2 == 0 and s or bad) n = n + (a or 0) + (b or 0)",
        "n = n + (utf8.offset(i % 3 == 0 and bad or s, 2) or 0)",
        "local a, b, c = utf8.codepoint(s, 1, i % 4 + 1) n = n + a + (b or 0) + (c or 0)",
        "n = n + #utf8.char(utf8.codepoint(s, 1, i % 4 + 1))",
        "n = n + #utf8.char(1, 2, 3, 4, 5, 6, 7, 8)",
    ];
    for body in cases {
        let setup = format!("{TEXT} local bad = 'a\\xffb'");
        agrees(&looped(&setup, body));
    }
}
