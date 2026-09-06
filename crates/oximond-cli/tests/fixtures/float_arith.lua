-- Float % and ^ edge cases and a hot loop, printed so that the last bit
-- of every result shows. Runs unchanged under Lua 5.1 to 5.5 and LuaJIT.
local inf, nan = 1 / 0, 0 / 0
local function show(x)
  if x ~= x then return "nan" end
  return string.format("%.17g", x)
end

-- Operands live in variables, so nothing is folded at compile time.
local pairs_ = {
  { -583, 3.001 }, { 1e10, 4 / -2.25 }, { 5.0, inf }, { -5.0, inf },
  { 5.0, -inf }, { -5.0, -inf }, { inf, 3.0 }, { -inf, 3.0 }, { 3.0, 0.0 },
  { 0.0, 3.0 }, { -0.0, 3.0 }, { 0.0, -3.0 }, { -0.0, -3.0 }, { 6.0, 3.0 },
  { -6.0, 3.0 }, { 6.0, -3.0 }, { nan, 2.0 }, { 2.0, nan }, { 5.5, -inf },
  { -0.0, -inf }, { inf, inf }, { 1e300, 1e-300 }, { -1e300, 7.7 },
  { 5.3, 2.0 }, { -5.3, 2.0 }, { 5.3, -2.0 }, { -1.5e-200, 1e-200 },
  { 1.5e-200, -1e-200 }, { 2.0 ^ 53 * 3 + 5, 1000003.0 }, { 0.1, 0.01 },
}
for index, pair in ipairs(pairs_) do
  local a, b = pair[1], pair[2]
  print("mod", index, show(a % b))
end

local squares = { 483 ^ 3, 1.1, 3.14159, 0.1, -7.77, 1e154, 123456.789, nan, inf, 0.0 }
for index, x in ipairs(squares) do
  local two = 2.0
  print("pow", index, show(x ^ 2), show(x ^ two), show(x ^ 3), show(x ^ 0.5))
end

-- Hot loop: float % and ^ stay in a trace once the loop is hot.
local function run(scale)
  local sum, last = 0.0, 0.0
  for i = 1, 3000 do
    local x = i * 0.37 * scale - 400.0
    local y = (i % 17) * 0.7 - 5.9
    if y == 0.0 then y = 1.25 end
    local m1 = x % y
    local m2 = (i - 583.0) % 3.001
    local m3 = 1e10 % -(i * 0.0001 + 1.7777)
    local p1 = (x * 1.0001) ^ 2
    local p2 = x ^ 2.0
    sum = sum + m1 + m2 * 3.0 + m3 + p1 * 1e-9 + p2 * 1e-9
    last = m1 + m2 + m3 + p1 + p2
    if i % 500 == 0 then print("loop", i, show(sum), show(last)) end
  end
  return sum
end
print("run", show(run(1.0)), show(run(-2.5)))

-- Tiny numeric callees and table fields take the interpreter's inline plans.
local function modulo(a, b) return a % b end
local function square(a) return a ^ 2 end
local function recur(n, acc) if n == 0 then return acc end return recur(n - 1, (acc * 1.7 + 3.3) % 7.1) end
local cell = { v = 1.5 }
local total = 0.0
for i = 1, 600 do
  local x = i * 1.37 - 300.0
  cell.v = (cell.v + x) % 9.7
  total = total + modulo(x, 3.001) + modulo(-x, -2.9) + square(x * 1.01) * 1e-9 + cell.v
  if i % 150 == 0 then print("plan", i, show(total), show(cell.v), show(recur(40, x))) end
end
