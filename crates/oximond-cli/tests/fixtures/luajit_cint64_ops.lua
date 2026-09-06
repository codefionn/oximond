-- Register-operand 64-bit integer cdata arithmetic, comparisons and bit ops
-- over a grid of operands. `luajit_cint64_ops.expected` is this script's
-- output under LuaJIT 2.1 on x64, which oximond matches row for row (see the
-- test).
local vals = {
  0ull, 1ull, 2ull, 0xffffffffffffffffull, 0x8000000000000000ull,
  0ll, 1ll, -1ll, 2ll, 0x7fffffffffffffffll, -0x7fffffffffffffffll - 1,
  0, 1, 2, -2, 0.5, -2.7, 2.7, 1e19, -1e19, 2^63, 2^64, 0/0,
}
local names = {}
for i = 1, #vals do names[i] = i end
local ops = {
  add = function(a, b) return a + b end,
  sub = function(a, b) return a - b end,
  mul = function(a, b) return a * b end,
  div = function(a, b) return a / b end,
  mod = function(a, b) return a % b end,
  pow = function(a, b) return a ^ b end,
  eq = function(a, b) return a == b end,
  lt = function(a, b) return a < b end,
  le = function(a, b) return a <= b end,
  gt = function(a, b) return a > b end,
  band = bit.band, bor = bit.bor, bxor = bit.bxor,
  lshift = bit.lshift, rshift = bit.rshift, arshift = bit.arshift,
  rol = bit.rol, ror = bit.ror,
}
local order = { "add", "sub", "mul", "div", "mod", "pow", "eq", "lt", "le", "gt",
  "band", "bor", "bxor", "lshift", "rshift", "arshift", "rol", "ror" }
local function is_cdata(v) return type(v) == "cdata" end
local function cd(v) return is_cdata(v) or type(v) ~= "number" end
for _, name in ipairs(order) do
  local f = ops[name]
  for i = 1, #vals do
    for j = 1, #vals do
      local a, b = vals[i], vals[j]
      if is_cdata(a) or is_cdata(b) then
        local ok, r = pcall(f, a, b)
        print(name, i, j, ok and type(r) or "error", ok and tostring(r) or "")
      end
    end
  end
end
for i = 1, #vals do
  local a = vals[i]
  if is_cdata(a) then
    local ok, r = pcall(bit.bnot, a)
    print("bnot", i, ok and type(r) or "error", ok and tostring(r) or "")
    ok, r = pcall(function() return -a end)
    print("neg", i, ok and type(r) or "error", ok and tostring(r) or "")
  end
end
local ok, r = pcall(bit.band, 1ull, 2ll, 3, 0xffffffffffffffffull)
print("band4", ok and type(r) or "error", ok and tostring(r) or "")
print("spec1", 0xffffffffffffffffull + 1 == 0ull, 1ull - 2, -1ll * 2, 1ull + 0.5, 1ll + 2ull)
for _, e in ipairs { "return 1ull / 0", "return 1ull % 0", "return 1ll / 0", "return 1ll % 0" } do
  local ok2, r2 = pcall(loadstring(e))
  print(e, ok2 and tostring(r2) or "error")
end
