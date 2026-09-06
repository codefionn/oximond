//! Version-specific library observations, bounded and independent of host state.

use crate::program::Node;
use std::fmt::Write as _;

impl super::Generator {
    pub(super) fn profile_library_probe(&mut self) -> Node {
        self.feature("profile_libraries");
        let tag = self.tag("L");
        let mut text = format!("do\n  local tag = {tag}\n");
        if self.dialect.table_pack {
            self.feature("library_table_pack_unpack");
            text.push_str("  local t = table.pack(false, nil, 7, nil)\n  out(tag, t.n, table.unpack(t, 1, t.n))\n  out(tag, rawlen({false, 7, 9}), rawlen('a\\000b'))\n");
        } else {
            self.feature("library_legacy_table");
            text.push_str("  local t = {4, 3, 2, 1}\n  out(tag, unpack(t, 2, 3), table.maxn(t))\n  local total = 0\n  table.foreachi(t, function(i, v) total = total + i * v end)\n  out(tag, total)\n");
        }
        if self.expanded_probes && self.dialect.lua55 {
            self.feature("library_table_create");
            text.push_str("  local created = table.create(3, 2)\n  out(tag, type(created), next(created), #created)\n  created[1], created[2], created.name = 11, false, 'made'\n  out(tag, created[1], created[2], created.name, #created)\n  for _, hints in ipairs({{-1, 0}, {1.5, 0}, {0, -1}, {'bad', 0}}) do\n    local ok, value = pcall(table.create, hints[1], hints[2])\n    out(tag, ok, type(value))\n  end\n");
        }
        if self.expanded_probes {
            self.feature("library_string_dump");
            text.push_str("  local dumped = string.dump(function(x) local y = x + 2; return y, y * 3 end)\n  local loaded = assert((loadstring or load)(dumped))\n  out(tag, loaded(7))\n");
            self.feature("library_string_dump_upvalues");
            text.push_str("  local first, second = 7, 9\n  local function captured(x) return first + x, second end\n  local restored = assert((loadstring or load)(string.dump(captured)))\n  local first_name = debug.setupvalue(restored, 1, 11)\n  local second_name = debug.setupvalue(restored, 2, 3)\n  out(tag, first_name ~= nil, second_name ~= nil, restored(5))\n  local dump_ok, dump_error = pcall(string.dump, print)\n  out(tag, dump_ok, type(dump_error))\n");
            if self.dialect.integers {
                self.feature("library_string_dump_strip");
                text.push_str("  local stripped = string.dump(function(x) return x * 2, x - 1 end, true)\n  local loaded_strip = assert(load(stripped))\n  out(tag, loaded_strip(9))\n");
            }
        }
        if self.dialect.integers {
            self.feature("library_integer_math");
            text.push_str("  out(tag, math.type(3), math.type(3.0), math.tointeger('12'), math.tointeger(1.5), math.ult(-1, 0))\n  out(tag, math.mininteger, math.maxinteger)\n");
            self.feature("library_string_pack");
            text.push_str("  local packed = string.pack('<i2I2c3', -321, 65000, 'abc')\n  out(tag, #packed, string.packsize('<i2I2c3'), string.unpack('<i2I2c3', packed))\n  local variable = string.pack('>s2z', 'a\\000b', 'tail')\n  out(tag, string.unpack('>s2z', variable))\n");
            if self.expanded_probes {
                self.feature("library_string_pack_variants");
                let (format, arguments, fixed) = *self.rng.pick(&[
                    ("<i1I1", "-12, 250", true),
                    (">!4 b I2 f d", "-5, 4095, 1.5, -2.25", true),
                    ("=c5s1z", "'ab', 'xy', 'end'", false),
                    ("<I4i2", "4294967295, -1234", true),
                    (">!8 I1 d i4", "200, 0.5, -123456", true),
                ]);
                let _ = write!(
                    text,
                    "  local fmt = '{format}'\n  local bytes = string.pack(fmt, {arguments})\n  out(tag, #bytes, string.unpack(fmt, bytes))\n  local prefixed = '12345678' .. bytes\n  out(tag, string.unpack(fmt, prefixed, 9))\n  local ok, value = pcall(string.unpack, fmt, string.sub(bytes, 1, #bytes - 1))\n  out(tag, ok, type(value))\n"
                );
                if fixed {
                    text.push_str("  out(tag, string.packsize(fmt))\n");
                }
                text.push_str("  local ok1, error1 = pcall(string.pack, 'i1', 128)\n  local ok2, error2 = pcall(string.unpack, 'i2', 'x')\n  out(tag, ok1, type(error1), ok2, type(error2))\n");
            }
            self.feature("library_table_move");
            text.push_str("  local t = {1, 2, 3, 4}\n  out(tag, table.move(t, 1, 3, 2) == t)\n  out(tag, table.unpack(t))\n  local target = {}\n  table.move(t, 2, 4, 1, target)\n  out(tag, table.unpack(target))\n");
        } else {
            self.feature("library_legacy_math");
            text.push_str("  local m, e = math.frexp(12)\n  out(tag, m, e, math.ldexp(m, e), math.pow(2, 5), math.log10(100))\n");
        }
        if self.dialect.bit32_library {
            self.feature("library_bit32");
            text.push_str("  out(tag, bit32.band(0xFEDC, 0xFF), bit32.bnot(0), bit32.extract(0xABCD, 4, 8), bit32.replace(0, 15, 4, 4))\n  out(tag, bit32.lrotate(1, 31), bit32.arshift(0x80000000, 31))\n");
        }
        if self.dialect.bit_library {
            self.feature("library_luajit_bit");
            text.push_str("  out(tag, bit.tobit(4294967295), bit.tohex(-1), bit.bswap(0x12345678), bit.rol(1, 31), bit.arshift(-2, 1))\n");
        }
        text.push_str("end");
        self.leaf(text)
    }

    pub(super) fn utf8_probe(&mut self) -> Node {
        self.feature("utf8");
        let tag = self.tag("U");
        let cp = *self.rng.pick(&[
            0,
            0x7F,
            0x80,
            0x7FF,
            0x800,
            0xD7FF,
            0xE000,
            0xFFFF,
            0x10000,
            0x0010_FFFF,
        ]);
        let mut text = format!(
            "do\n  local tag = {tag}\n  local s = utf8.char(0, 65, {cp}, 0x20AC, 0x1F642)\n"
        );
        self.feature("utf8_roundtrip");
        text.push_str("  out(tag, #s, utf8.len(s), utf8.codepoint(s, 1, -1))\n  for p, c in utf8.codes(s) do out(tag, p, c, utf8.offset(s, 0, p), utf8.offset(s, 1, p)) end\n  for n = -6, 6 do out(tag, n, utf8.offset(s, n)) end\n  out(tag, utf8.len(s, 2, -1), utf8.codepoint(s, -4, -1))\n  local count = 0\n  for chunk in string.gmatch(s, utf8.charpattern) do count = count + 1; out(tag, #chunk) end\n  out(tag, count)\n");
        self.feature("utf8_iterator_control");
        text.push_str("  local iterator, state, control = utf8.codes(s)\n  out(tag, type(iterator), state == s, control)\n  for i = 1, 6 do\n    local p, c = iterator(state, control)\n    out(tag, p, c)\n    if not p then break end\n    control = p\n  end\n  out(tag, iterator(state, 1))\n  out(tag, iterator(state, 3))\n");
        self.feature("utf8_partial_range");
        text.push_str("  local lead = #s - 3\n  out(tag, utf8.len(s, lead, lead), utf8.codepoint(s, lead, lead))\n  out(tag, utf8.len(s, 3, 3), utf8.codepoint(s, 3, 3))\n");
        self.feature("utf8_invalid");
        text.push_str("  local invalid = {'\\128', '\\192\\128', '\\226\\130', '\\244\\144\\128\\128', 'a\\255b', '\\226x\\172'}\n  for i = 1, #invalid do\n    local bad = invalid[i]\n    out(tag, i, utf8.len(bad))\n    local ok, err = pcall(utf8.codepoint, bad, 1, -1)\n    out(tag, ok, type(err))\n    ok, err = pcall(function() for p, c in utf8.codes(bad) do out(tag, p, c) end end)\n    out(tag, ok, type(err))\n  end\n  for _, args in ipairs({{s, 1, 4}, {s, 1, 0}, {s, 0, #s + 2}}) do\n    local ok, err = pcall(utf8.offset, args[1], args[2], args[3])\n    out(tag, ok, type(err))\n  end\n  for _, c in ipairs({-1, 2147483648, 1.5}) do local ok, err = pcall(utf8.char, c); out(tag, ok, type(err)) end\n");
        if self.dialect.utf8_lax {
            self.feature("utf8_lax");
            text.push_str("  local extended = utf8.char(0xD800, 0x110000, 0x7FFFFFFF)\n  out(tag, utf8.len(extended), utf8.len(extended, 1, -1, true))\n  out(tag, utf8.codepoint(extended, 1, -1, true))\n  for p, c in utf8.codes(extended, true) do out(tag, p, c) end\n");
        }
        text.push_str("end");
        self.leaf(text)
    }
}
