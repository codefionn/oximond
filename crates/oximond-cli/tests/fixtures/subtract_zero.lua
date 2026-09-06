local function f(x) return string.format("%.17g", x) end
local z = -0.0
local nz = -0.0
print(f(z - 0), f((-0.0) - 0), f(z - 0.0), f(z + 0), f(z + 0.0), f(0 - z), f(0.0 - z), f(z - 1 + 1))
print(f(z - 0.0), f(z - -0), f(z + -0), f(z - 127), f(z - 128), f(z - -127), f(z - -128))
print(f(-0.0 - 0), f(-0.0 + 0), f(-0.0 - 0.0), f(0 - 0.0), f(-(0.0) - 0))
print(f(z - 0x0), f(z - (0)), f(z - 1e0), f(z-(1-1)))
print(f(z * 1), f(z // 1), f(z % 1), f(-z - 0))
local t = {z}
print(f(t[1] - 0), f(t[1] + 0), f(t[1] - 0.0))
local function g(a) return a - 0, a + 0, a - 0.0, 0 - a, a - 1, a + (-0) end
print(f(g(z)), f(select(2, g(z))), f(select(3, g(z))), f(select(4, g(z))))
for i = 1, 3 do local r = z - 0; io.write(f(r), " ") end print()
local function h(a, n) local r = 0; for i=1,n do r = a - 0 end return r end
print(f(h(z, 100000)))
local w = 5 - 5
print(f(-0.0 - w), f(z - w), f(z - (2-2)), f(z - 2^0 + 1))
print(f(z - 0xFFFF), f(z - 65535), f(z - 255), f(z-256), f(z - 32767), f(z - 32768))
print(f(z - 65536))
local mz = -0.0
local k = 0
print(f(mz - k))
local mt = {__sub = function(a, b) return "sub", b end, __add = function() return "add" end}
local t = setmetatable({}, mt)
print(t - 0)
print(pcall(function() local n; return n - 0 end))
print(pcall(function() return {} - 0 end))
print("-0.0" - 0, "5" - 0, math.type("5" - 0), "0" - 0, math.type(5 - 0), 5 - 0, 5.5 - 0, math.type(0 - 0))
print(1/(0.0 - 0), 1/(-0.0 - 0), 1/(0 - 0), 0/0 - 0)
local x = -0.0
local y = x - 0
local u; u = x - 0
local tt = {x - 0, [1+1] = x - 0}
print(1/y, 1/u, 1/tt[1], 1/tt[2])
local function f2(a) return a - 0 end
print(1/f2(-0.0), f2(3), f2(math.mininteger), f2(2^63))
for i = 1, 200 do assert(1/f2(-0.0) == math.huge) end
print(1/(x - (0)), 1/(x - -0), 1/(x - (3-3)), 1/(x - 0*5), 1/(x - 0.0))

-- Folded <const> locals and operators pick OP_ADDI like a literal does.
do
local z = -0.0
local k <const> = 0
print(f(z - k), f(z - (k)), f(z - (k + 0)))
local q <const> = 1 - 1
print(f(z - q), f(z - (2 // 2 - 1)), f(z - (0 | 0)))
local a <const>, b <const> = 0, 0
print(f(z - a), f(z - b))
local c <const> = 0
local function up() return z - c end
print(f(up()))
do local c = 0; print(f(z - c)) end
local d <const> = 5 % 5
print(f(z - d))
print(f(z - (3 & 0)), f(z - (~(-1))), f(z - (1 << 64)), f(z - (1 >> 0 >> 1)), f(z - (-1 >> 64)), f(z - (5 ~ 5)), f(z - ~-1))
print(f(z - (-0)), f(z - (1 << -64)), f(z - (0 * 3)), f(z-(0//3)), f(z-(-0//3)))
local e <const> = 1.0
print(f(z - (e - 1)))
local s <const> = "x"
local m <const> = 0
local n <const> = m + 0
print(f(z - n))
local g <const> = k
print(f(z - g))
local h <const> = 1 - 1
local function u2() local h <const> = h; return z - h end
print(f(u2()))
print(f(z - (1.0//1 - 1)), f(z - (2^0 - 1)), f(z - (4 // 2 - 2)))
print(f(z - (0 // 1 // 1)))
end
do
local z = -0.0
local k <const> = 0
local k = 0
print(f(z - k))
local j <const> = 0
do local j = 0; print(f(z - j)) end
local function w(j) return z - j end
print(f(w(0)))
end
