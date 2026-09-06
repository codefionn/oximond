//! `tonumber`, `string.pack` and `string.unpack` in hot loops. A compiled
//! trace answers these on the bytes it holds, without the native's argument
//! and result lists, so each program runs under every profile in both
//! execution modes and checks itself with `assert`: the answers on a trace
//! are the interpreter's, for the calls the direct paths take and for the
//! ones that must leave them.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

const PROFILES: [CompatibilityProfile; 6] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua52,
    CompatibilityProfile::Lua53,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::Lua55,
    CompatibilityProfile::LuaJit21,
];

fn run(source: &str) {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .profile(profile)
                .execution_mode(mode)
                .build()
                .unwrap();
            lua.vm_mut().use_synchronous_jit_compilation();
            if let Err(error) = lua.exec(source) {
                panic!("{profile} {mode}: {error:?}");
            }
        }
    }
}

#[test]
fn tonumber_of_strings_in_a_loop_matches_single_calls() {
    run(r#"
        local convert = tonumber
        local decimal = { "0", "17", "-29", "304", "8125", "-7", "49", "1001" }
        local hex = { "ff", "2a", "7f", "10", "abc", "dead", "0", "1e" }
        local hex_values = { 255, 42, 127, 16, 2748, 57005, 0, 30 }
        local invalid = { "bad!", "1x", "--2", "3.4.5", "", ".", "1 2", "0x" }
        local total, misses = 0, 0
        for i = 1, 4000 do
            local at = (i - 1) % 8 + 1
            local d, h = decimal[at], hex[at]
            local plain, ten, sixteen = convert(d), convert(d, 10), convert(h, 16)
            assert(plain == ten and sixteen == hex_values[at], h)
            total = total + plain + sixteen
            if convert(invalid[at]) == nil then
                misses = misses + 1
            end
        end
        assert(total == 500 * (9460 + 60223), total)
        assert(misses == 4000, misses)

        -- One call site sees whole numbers, fractions, exponents,
        -- hexadecimal numerals and strings that are no number.
        local mixed = { "12", "3.25", "1e2", "0x10", " 7 ", "5.", ".5", "-0.5", "nope", "+3" }
        local wanted = { 12, 3.25, 100, 16, 7, 5, 0.5, -0.5, false, 3 }
        for i = 1, 4000 do
            local at = (i - 1) % 10 + 1
            local got = convert(mixed[at])
            if wanted[at] then
                assert(got == wanted[at], mixed[at])
            else
                assert(got == nil, mixed[at])
            end
        end
        if math.type then
            for i = 1, 3000 do
                assert(math.type(convert("17")) == "integer")
                assert(math.type(convert("17.0")) == "float")
                assert(math.type(convert("ff", 16)) == "integer")
            end
        end

        -- Calls the direct path leaves to the native: a number, a base that
        -- is a float or a string, a base out of range.
        local bases = { 16, 16.0, "16" }
        for i = 1, 3000 do
            assert(convert(i) == i)
            assert(convert(i + 0.5) == i + 0.5)
            assert(convert("ff", bases[i % 3 + 1]) == 255)
            assert(convert("zz", 36) == 1295)
            assert(convert("12", 2) == nil)
            assert(convert("-101", 2) == -5 or convert("-101", 2) > 2 ^ 63)
        end
        for i = 1, 300 do
            local ok = pcall(convert, "1", i % 2 == 0 and 37 or 1)
            assert(not ok)
        end
    "#);
}

#[test]
fn string_pack_and_unpack_in_a_loop_match_single_calls() {
    run(r#"
        if not string.pack then
            return
        end
        local pack, unpack = string.pack, string.unpack
        local words = { "naive", "cafe", "", "ascii", "a much longer name than fits inline" }
        local total = 0
        for round = 1, 4000 do
            local name = words[round % #words + 1]
            local record = pack("<I4I2ds1", round, round % 1000, round / 8, name)
            assert(#record == 15 + #name)
            local id, flags, weight, after = unpack("<I4I2d", record)
            assert(id == round and flags == round % 1000 and weight == round / 8 and after == 15)
            local label, finish = unpack("<s1", record, after)
            assert(label == name and finish == #record + 1)
            -- Big-endian and signed items, read back from the middle.
            local signed = pack(">i2i4i8b", -round, round - 2000, -round * 3, round % 100 - 50)
            local a, b, c, d, stop = unpack(">i2i4i8b", signed)
            assert(a == -round and b == round - 2000 and c == -round * 3 and d == round % 100 - 50)
            assert(stop == 16)
            local c_again = unpack(">i8", signed, 7)
            assert(c_again == c)
            -- A negative start counts from the end.
            assert(unpack("b", signed, -1) == d)
            total = total + id + #label
        end
        assert(total > 0)

        -- A format and five values: the most arguments a trace hands over.
        for i = 1, 3000 do
            local five = pack("<I4I2I2I1I1", i, i % 100, 7, 1, 2)
            assert(#five == 10)
            local a, b, c, d, e = unpack("<I4I2I2I1I1", five)
            assert(a == i and b == i % 100 and c == 7 and d == 1 and e == 2)
        end

        -- Calls that raise an error: a value out of range, data too short, a
        -- start outside the string.
        for i = 1, 600 do
            assert(not pcall(pack, "<I1", 255 + i))
            assert(not pcall(pack, "<i1", -129 - i))
            assert(not pcall(unpack, "<I4", "abc"))
            assert(not pcall(unpack, "<I4", "abcdefgh", 7))
            assert(not pcall(unpack, "<I1", "abc", 5 + i))
            assert(pack("<i1", i % 100 - 50):byte() == (i % 100 - 50) % 256)
        end

        -- Arguments the direct path leaves to the native: a float that is
        -- an integer, a numeric string, a float start.
        for i = 1, 3000 do
            assert(pack("<i2", i + 0.0) == pack("<i2", i))
            assert(pack("<i2", "258") == "\2\1")
            assert(unpack("<I2", "\0\2\1", 2.0) == 258)
            assert(unpack("<I2", "\0\2\1", "2") == 258)
        end
    "#);
}

#[test]
fn utf8_char_in_a_loop_builds_short_and_long_strings() {
    run(r#"
        if not utf8 then
            return
        end
        local char, codepoint, len = utf8.char, utf8.codepoint, utf8.len
        -- The widest character the profile encodes: four bytes in Lua 5.3,
        -- six after it.
        local widest = pcall(char, 0x7FFFFFFF) and 0x7FFFFFFF or 0x10FFFF
        local width = #char(widest)
        local lax = widest > 0x10FFFF or nil
        for i = 1, 4000 do
            local one = char(i)
            assert(codepoint(one) == i and len(one) == 1)
            local three = char(i + 1, 0x20AC, 65)
            local a, b, c = codepoint(three, 1, -1)
            assert(a == i + 1 and b == 0x20AC and c == 65)
            assert(three:sub(-4) == "\226\130\172A")
            -- Five wide characters are longer than a string keeps inline.
            local five = char(widest, widest - 1, 0x10000, i + 0x800, widest)
            assert(#five == 3 * width + 4 + 3, #five)
            local first, _, _, fourth, last = codepoint(five, 1, -1, lax)
            assert(first == widest and fourth == i + 0x800 and last == widest)
        end
        assert(char() == "")
        for i = 1, 300 do
            assert(not pcall(char, -i))
            assert(not pcall(char, widest + i))
        end
    "#);
}

#[test]
fn table_concat_in_a_loop_joins_short_and_long_elements() {
    run(r#"
        local total = 0
        for round = 1, 60 do
            local parts = {}
            for i = 1, 200 do
                parts[i] = tostring(i % 101)
            end
            if round % 7 == 0 then
                parts[50] = string.rep("x", 40)
            end
            local joined = table.concat(parts, ",")
            local expected = 0
            for i = 1, 200 do
                expected = expected + #parts[i]
            end
            assert(#joined == expected + 199, #joined)
            assert(joined:sub(1, 8) == "1,2,3,4,", joined:sub(1, 8))
            assert(joined:sub(-6) == ",98,99", joined:sub(-6))
            total = total + #table.concat(parts) + #table.concat(parts, "<->")
        end
        assert(total > 0)
    "#);
}
