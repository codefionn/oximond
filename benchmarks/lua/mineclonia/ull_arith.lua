-- Arithmetic that mixes LuaJIT 64-bit integers with plain numbers, as the
-- helpers in mcl_levelgen/random.lua do when the runtime has `1ull'
-- literals: `0x100000000ull * a[2] + a[1]', `a[1] * 1ull',
-- `(UINT_MAX - lohi) - hi'. The kernel is compiled once with the
-- 64-bit literals and, on runtimes without them, once more with plain
-- numbers. Every value stays below 2^53, so both versions agree.
local scale = tonumber(arg and arg[1]) or 1

local source = [[
    local fmod = math.fmod
    return function(count)
        local acc = 0ull
        local halves = { 0, 0 }
        for i = 1, count do
            halves[1] = (i * 7919) % 1048576
            halves[2] = i % 1024
            local along = 0x100000ull * halves[2] + halves[1]
            local blong = along * 3ull + 12345
            local m1 = halves[1] * 1ull
            local diff = (0xffffffffull - blong) - m1
            acc = (acc + blong + diff % 65536ull) % 1000000007ull
            if along < blong then
                acc = acc + 1
            end
        end
        return acc
    end
]]

local kernel = loadstring and loadstring(source)
if not kernel then
    -- Lua 5.1 and other runtimes without 64-bit integer literals.
    kernel = assert(loadstring((source:gsub("(%x)ull", "%1"))))
end
local result = kernel()(50000 * scale)
print(tonumber(result) or result)
