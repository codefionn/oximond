-- Binary records and UTF-8 text. The first loop packs a record, unpacks it
-- again and takes the name apart with the utf8 library (length, first and
-- last code point, a character rebuilt with utf8.char). The second walks a
-- long text with utf8.codes and re-encodes what it reads. Lua 5.3 and later
-- only: string.pack and the utf8 library do not exist before.
local scale = tonumber(arg and arg[1]) or 1

local pack, unpack = string.pack, string.unpack
local len, codepoint, char, offset, codes = utf8.len, utf8.codepoint, utf8.char, utf8.offset, utf8.codes

local words = { "na\u{EF}ve", "caf\u{E9}", "\u{65E5}\u{672C}\u{8A9E}", "ascii", "se\u{F1}or", "\u{20AC}uro", "\u{1F600}ok" }
local count = #words

local total = 0
for round = 1, 30000 * scale do
    local name = words[round % count + 1]
    local record = pack("<I4I2ds1", round, round % 1000, round / 8, name)
    local id, flags, weight, after = unpack("<I4I2d", record)
    local label, finish = unpack("<s1", record, after)
    local length = len(label)
    local first = codepoint(label, 1)
    local last = codepoint(label, offset(label, -1))
    local shifted = char(first + 1, last, 0x20AC)
    total = (total + id + flags + length + first + last + finish + #shifted + #record + weight // 1) % 1000000007
end

local text = table.concat(words, " "):rep(60)
local sum = 0
for pass = 1, 40 * scale do
    for position, code in codes(text) do
        sum = (sum + code * 3 + position) % 1000000007
    end
    sum = sum + len(text)
end
print(total, sum)
