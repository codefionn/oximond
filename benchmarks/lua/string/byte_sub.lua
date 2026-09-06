local scale = tonumber(arg and arg[1]) or 1
local byte, sub = string.byte, string.sub
local source = string.rep("0123456789abcdef", 16)

-- Rotate slices so byte and sub must use changing offsets and live results.
local total = 0
for i = 1, 180000 * scale do
    local offset = (i - 1) % 253 + 1
    local slice = sub(source, offset, offset + 3)
    local a, b, c, d = byte(slice, 1, 4)
    total = total + a + b + c + d + byte(source, offset)
end
print(total)
