-- Library calls whose results feed another call or a return, as in
-- mcl_levelgen/random.lua: `a[1] = tonumber (band (value, 0xffffffffull))',
-- `string.byte (str:sub (i, i))' in `stringtoull', and helpers that end in
-- `return bor (fn (32), 0)' or `return tonumber (excess)'. The inner call
-- has an open result count and the outer one takes all of its results.
local scale = tonumber(arg and arg[1]) or 1

local floor, abs, max, min = math.floor, math.abs, math.max, math.min
local byte, sub = string.byte, string.sub
local tonumber = tonumber

-- Ends in a call of a library function.
local function low_word(x)
    return floor(x % 4294967296)
end

local function split(a, value)
    a[1] = tonumber(floor(value / 65536))
    a[2] = abs(floor(value % 65536))
    return low_word(a[1] * 3 + a[2])
end

local function digits(str)
    local total = 0
    for i = 1, #str do
        total = total * 10 + max(min(byte(sub(str, i, i)) - 48, 9), 0)
    end
    return total
end

local function run(count)
    local a = { 0, 0 }
    local sum = 0
    local text = "3948575739"
    for i = 1, count do
        sum = (sum + split(a, i * 7919 + sum)) % 1000000007
        if i % 32 == 0 then
            sum = (sum + digits(text)) % 1000000007
        end
    end
    return sum
end

print(run(24000 * scale))
