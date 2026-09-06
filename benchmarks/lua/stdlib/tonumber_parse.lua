local scale = tonumber(arg and arg[1]) or 1
local convert = tonumber
local decimal = { "0", "17", "-29", "304", "8125", "-7", "49", "1001" }
local hex = { "ff", "2a", "7f", "10", "abc", "dead", "0", "1e" }
local invalid = { "bad!", "1x", "--2", "3.4.5" }

-- Parse prebuilt valid and invalid strings, with explicit decimal and hex bases.
local total = 0
for i = 1, 100000 * scale do
    local d = decimal[(i - 1) % 8 + 1]
    local h = hex[(i - 1) % 8 + 1]
    local bad = invalid[(i - 1) % 4 + 1]
    total = total + convert(d) + convert(d, 10) + convert(h, 16)
    if convert(bad) == nil then
        total = total + 1
    end
end
print(total)
