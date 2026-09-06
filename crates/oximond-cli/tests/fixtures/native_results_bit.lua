-- bit.* on numbers, 64-bit integer cdata and mixes of both, called often
-- enough in loops that the interpreter's scalar native path and the trace
-- recorder both see them. Prints the type and text of every result.
local bit = require("bit")
local band, bor, bxor, bnot = bit.band, bit.bor, bit.bxor, bit.bnot
local lshift, rshift, arshift = bit.lshift, bit.rshift, bit.arshift
local rol, ror, bswap, tobit = bit.rol, bit.ror, bit.bswap, bit.tobit

local function show(...)
    local out = {}
    for i = 1, select("#", ...) do
        local v = select(i, ...)
        out[#out + 1] = type(v) .. ":" .. tostring(v)
    end
    print(table.concat(out, " "))
end

local numbers = {
    0, 1, -1, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 255, 65535, 4294967295,
    4294967296, 4294967297, 2147483647, 2147483648, -2147483648, -2147483649,
    1e10, -1e10, 2 ^ 50, -2 ^ 50, 1e15, -1e15, 2251799813685247.5,
    0.49999999999999994, 4294967295.5,
}
local wide = { 0ll, 1ll, -1ll, 0xffffffffll, 0x100000000ll, 0x7fffffffffffffffll,
    -0x7fffffffffffffffll - 1, 0ull, 1ull, 0xffffffffull, 0x100000000ull,
    0xffffffffffffffffull, 0x8000000000000000ull }

for _, a in ipairs(numbers) do
    show(a, tobit(a), bnot(a), bswap(a), band(a, 0xff), bor(a, 1), bxor(a, 0x55))
end

for _, a in ipairs(numbers) do
    for _, count in ipairs { 0, 1, 5, 31, 32, 33, 63, 64, 65, -1, 1.5, 2 ^ 32 + 3 } do
        show(a, count, lshift(a, count), rshift(a, count), arshift(a, count),
            rol(a, count), ror(a, count))
    end
end

for _, a in ipairs(wide) do
    show(a, tobit(a), bnot(a), bswap(a))
    for _, b in ipairs(wide) do
        show(a, b, band(a, b), bor(a, b), bxor(a, b))
    end
    for _, b in ipairs { 0, 1, 7, 31, 32, 33, 63, 64, 65, -1, 2.5, 4294967296 } do
        show(a, b, band(a, b), bor(b, a), bxor(a, b), lshift(a, b), rshift(a, b),
            arshift(a, b), rol(a, b), ror(a, b))
    end
end

for _, a in ipairs(numbers) do
    show(a, band(a, 0x7ffffffffll), bor(0x10000000000ull, a), bxor(a, -1ll))
end

-- Strings coerce to numbers, and the first or last argument may be missing.
show(band("12", "10"), bor("0x10", 1), tobit("7"), bxor(3, "5"))
show(band(1, 2, 3, 7), bor(1, 2, 4, 8, 16), bxor(1, 3, 7, 15), band(5))
-- Error messages name the function differently per engine; only the
-- failure is compared.
show((pcall(band)))
show((pcall(lshift, 1)))
show((pcall(band, 1, {})))
show((pcall(band, "x", 1)))
show((pcall(tobit, nil)))
show((pcall(bnot, true)))
show((pcall(bswap)))

-- Hot loops over the fixed-count shapes, results in locals and as open
-- arguments of another call.
local sum, wsum = 0, 0ull
for i = 1, 3000 do
    local lo = band(i * 2654435, 0xffff)
    local hi = rshift(i * 2654435, 16)
    local mix = bxor(lshift(hi, 3), lo)
    sum = sum + tonumber(band(mix, 0xfff)) + tobit(bnot(mix))
    local w = 0x100000000ull * i + lo
    wsum = wsum + band(w, 0xffffffffull) + rshift(w, 32) + lshift(w, i % 70)
    sum = sum + tonumber(arshift(-w, i % 40))
end
show(sum, wsum)

-- A value that changes type inside a hot loop.
local acc = {}
for i = 1, 400 do
    local v = i
    if i % 5 == 0 then v = v * 1ll end
    if i % 7 == 0 then v = tostring(i) end
    if i % 11 == 0 then v = v + 0.5 end
    acc[#acc + 1] = tostring(band(v, 0xff)) .. "/" .. tostring(lshift(v, 4))
end
print(table.concat(acc, ","))
