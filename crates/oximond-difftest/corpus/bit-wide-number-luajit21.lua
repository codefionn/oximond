-- oximond-difftest repro profile=luajit21
-- LuaJIT converts number operands by adding 2^52 + 2^51 and taking
-- the low 32 bits of the resulting double, including outside +/-2^51.
local values = {
  3002399751580330, -3002399751580330,
  2251799813685247.5, 2251799813685248, 2251799813685249,
  -2251799813685248, -2251799813685249,
  4503599627370495, 4503599627370496, 4503599627370497,
  -4503599627370495, -4503599627370496, -4503599627370497,
  9007199254740991, 9007199254740992,
  -9007199254740991, -9007199254740992,
  1e20, -1e20, 1e300, -1e300,
  "3002399751580330",
}
for i, x in ipairs(values) do
  print(i, bit.tobit(x), bit.rshift(x, 3), bit.tohex(x), bit.bnot(x))
end
local result = 0
for i = 1, 3000 do
  local x = values[(i - 1) % #values + 1]
  result = bit.bxor(result, bit.tobit(x), bit.rshift(x, 3))
end
print("hot", result)
