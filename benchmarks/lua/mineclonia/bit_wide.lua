-- `bit' library calls with LuaJIT 64-bit integer arguments, as in the
-- helpers of mcl_levelgen/random.lua: `band (value, 0xffffffffull)',
-- `rshift (value, 32)', `lshift (along, k)', `band (a[1] * 1ull, b[1] * 1ull)'.
-- LuaJIT widens these operations to 64 bits. Each result goes to a local,
-- so the calls keep fixed result counts. Runtimes without 64-bit literals
-- run the same kernel on plain numbers with arithmetic stand-ins for the
-- four operations; every value stays below 2^53, so both versions agree.
local scale = tonumber(arg and arg[1]) or 1

local source = [[
    local band, bor, rshift, lshift = ...
    return function(count)
        local acc = 0ull
        for i = 1, count do
            local value = 0x100000000ull * (i % 1024) + (i * 2654435) % 4294967296
            local lo = band(value, 0xffffffffull)
            local hi = rshift(value, 32)
            local mid = band(rshift(value, 16), 0xffffull)
            local joined = bor(lshift(hi, 32), lo)
            local low12 = band(joined, 0xfffull)
            acc = acc + band(lo, 0xffull) + hi + mid + low12
        end
        return acc
    end
]]

local kernel = loadstring and loadstring(source)
local band, bor, rshift, lshift
if kernel and bit then
    band, bor, rshift, lshift = bit.band, bit.bor, bit.rshift, bit.lshift
else
    -- Exact for operands below 2^53, masks of the form 2^k - 1, and
    -- `bor' operands without common bits, which is all this kernel uses.
    local floor = math.floor
    kernel = assert(loadstring((source:gsub("(%x)ull", "%1"))))
    band = function(value, mask) return value % (mask + 1) end
    bor = function(a, b) return a + b end
    rshift = function(value, count) return floor(value / 2 ^ count) end
    lshift = function(value, count) return value * 2 ^ count end
end
local result = kernel(band, bor, rshift, lshift)(50000 * scale)
print(tonumber(result) or result)
