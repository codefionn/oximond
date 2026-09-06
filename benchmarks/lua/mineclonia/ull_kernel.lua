-- The 48-bit LCG of module_calls.lua with the 64-bit helpers that
-- mcl_levelgen/random.lua loads when the runtime has LuaJIT's `1ull'
-- literals: each helper widens its { lo, hi } halves to a 64-bit integer
-- (`0x100000000ull * a[2] + a[1]'), operates with the `bit' library and
-- splits the result again with `tonumber (band (value, 0xffffffffull))'.
-- Runtimes without 64-bit integers run the plain arithmetic helpers.
local scale = tonumber(arg and arg[1]) or 1

local floor = math.floor
local TWO32 = 4294967296
local TWO16 = 65536

-- a = a * m mod 2^64, for m < 2^32, in 16-bit limbs.
local function mulull(a, m)
    local lo, hi = a[1], a[2]
    local m0, m1 = m % TWO16, floor(m / TWO16)
    local l0, l1 = lo % TWO16, floor(lo / TWO16)
    local h0, h1 = hi % TWO16, floor(hi / TWO16)
    local c0 = l0 * m0
    local c1 = l0 * m1 + l1 * m0 + floor(c0 / TWO16)
    local c2 = l1 * m1 + h0 * m0 + floor(c1 / TWO16)
    local c3 = h0 * m1 + h1 * m0 + floor(c2 / TWO16)
    a[1] = (c1 % TWO16) * TWO16 + c0 % TWO16
    a[2] = (c3 % TWO16) * TWO16 + c2 % TWO16
end

local function shlull32(a)
    a[2] = a[1]
    a[1] = 0
end

local function addull(a, b)
    local lo = a[1] + b[1]
    local carry = 0
    if lo >= TWO32 then
        lo = lo - TWO32
        carry = 1
    end
    a[1] = lo
    a[2] = (a[2] + b[2] + carry) % TWO32
end

local function addkull(a, k)
    local lo = a[1] + k
    local carry = 0
    if lo >= TWO32 then
        lo = lo - TWO32
        carry = 1
    end
    a[1] = lo
    a[2] = (a[2] + carry) % TWO32
end

-- The mask's low half is all ones and its high half is 2^k - 1.
local function andull(a, mask)
    a[2] = a[2] % (mask[2] + 1)
end

-- Shift right a value below 2^53, whose sign bit is clear.
local function shrull(a, k)
    local value = floor((a[2] * TWO32 + a[1]) / 2 ^ k)
    a[1] = value % TWO32
    a[2] = floor(value / TWO32)
end

local wide = loadstring and loadstring([[
    local band, rshift, lshift, arshift = bit.band, bit.rshift, bit.lshift, bit.arshift
    local tonumber = tonumber

    local function mulull(a, m)
        local m1 = m * 1ull
        local lo = a[1] * m1
        local hi = a[2] * m1
        local lohi = rshift(lo, 32)
        hi = band(hi + lohi, 0xffffffffull)
        a[1] = tonumber(band(lo, 0xffffffffull))
        a[2] = tonumber(hi)
    end

    local function shlull32(a)
        local along = 0x100000000ull * a[2] + a[1]
        local value = lshift(along, 32)
        a[1] = tonumber(band(value, 0xffffffffull))
        a[2] = tonumber(rshift(value, 32))
    end

    local function addull(a, b)
        local along = 0x100000000ull * a[2] + a[1]
        local blong = 0x100000000ull * b[2] + b[1]
        local value = along + blong
        a[1] = tonumber(band(value, 0xffffffffull))
        a[2] = tonumber(rshift(value, 32))
    end

    local function addkull(a, k)
        local along = 0x100000000ull * a[2] + a[1]
        local value = along + k
        a[1] = tonumber(band(value, 0xffffffffull))
        a[2] = tonumber(rshift(value, 32))
    end

    local function andull(a, b)
        a[1] = tonumber(band(a[1] * 1ull, b[1] * 1ull))
        a[2] = tonumber(band(a[2] * 1ull, b[2] * 1ull))
    end

    local function shrull(a, k)
        local along = 0x100000000ull * a[2] + a[1]
        local value = arshift(along, k)
        a[1] = tonumber(band(value, 0xffffffffull))
        a[2] = tonumber(rshift(value, 32))
    end

    return mulull, shlull32, addull, addkull, andull, shrull
]])
if wide and bit then
    mulull, shlull32, addull, addkull, andull, shrull = wide()
end

local SEED_MASK = { 0xffffffff, 0xffff }

local function jvm_lcg(seed0)
    local seed, tmp = { seed0 % TWO32, 0 }, { 0, 0 }
    return function(nbits)
        tmp[1], tmp[2] = seed[1], seed[2]
        mulull(tmp, 0x5)
        shlull32(tmp)
        mulull(seed, 0xdeece66d)
        addull(seed, tmp)
        addkull(seed, 11)
        andull(seed, SEED_MASK)
        tmp[1], tmp[2] = seed[1], seed[2]
        shrull(tmp, 48 - nbits)
        return tmp[1]
    end
end

local function run(count)
    local next_bits = jvm_lcg(304853)
    local sum = 0
    for _ = 1, count do
        sum = (sum + next_bits(31)) % 1000000007
    end
    return sum
end

print(run(4000 * scale))
