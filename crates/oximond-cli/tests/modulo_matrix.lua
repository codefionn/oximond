-- Every remainder of two numbers a trace can meet: each sign, each subtype
-- pair, zero divisors, infinities, NaN and the integers at the ends of the
-- range. Each loop first runs long enough on ordinary operands to be
-- compiled, then meets the whole table, so the odd operands arrive in
-- compiled code and leave its fast paths. `mir_backend.rs` runs this under
-- every profile and compares the output with the interpreter's.

local kind = math.type or function() return "number" end
local compile = loadstring or load
local huge = math.huge
local nan = huge - huge
local WARM = 3000

local function show(v)
  if type(v) ~= "number" then return tostring(v) end
  if v ~= v then return "nan" end
  if kind(v) == "integer" then return "i" .. string.format("%d", v) end
  return string.format("%.17g", v)
end

local ints = {
  0, 1, -1, 2, -2, 7, -7, 13, -13, 101, -101, 4096, -4096,
  9007199254740991, -9007199254740991, 9007199254740993, -9007199254740993,
  9223372036854775807, -9223372036854775807 - 1,
}
local floats = {
  0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 3.5, -3.5, 7.0, -7.0, 0.1, -0.1,
  101.0, -101.0, 1e300, -1e300, 5e-324, -5e-324, 9007199254740992.0,
  -9007199254740994.0, 1.5e16, -1.5e16, huge, -huge, nan,
}
local both = {}
for i = 1, #ints do both[#both + 1] = ints[i] end
for i = 1, #floats do both[#both + 1] = floats[i] end

-- The operands of one loop: `WARM` ordinary pairs made by `ordinary`, then
-- every pair of `left` and `right`.
local function operands(left, right, ordinary)
  local xs, ys = {}, {}
  for i = 1, WARM do xs[i], ys[i] = ordinary(i) end
  for a = 1, #left do
    for b = 1, #right do
      xs[#xs + 1], ys[#ys + 1] = left[a], right[b]
    end
  end
  return xs, ys
end

local at = 0
local function remainders(xs, ys, out, from, to)
  for i = from, to do
    at = i
    out[i] = xs[i] % ys[i]
  end
end

-- Runs the loop over all of `xs`, going on after each pair that raises (an
-- integer remainder by zero does, where there are integers).
local function report(name, xs, ys, loop)
  local out, from, n = {}, 1, #xs
  while from <= n do
    at = from
    local ok = pcall(loop, xs, ys, out, from, n)
    if ok then break end
    out[at] = "error"
    from = at + 1
  end
  local sum, shown = 0, {}
  for i = 1, WARM do sum = sum + out[i] end
  for i = WARM + 1, n do shown[#shown + 1] = show(out[i]) end
  print(name, show(sum), table.concat(shown, " "))
end

local sign = { 1, -1 }
local pairs_of = {
  { "integer % integer", ints, ints, function(i)
    return sign[i % 2 + 1] * (i * 13), sign[i % 3 % 2 + 1] * (i % 9 + 2)
  end },
  { "float % float", floats, floats, function(i)
    return sign[i % 2 + 1] * (i * 0.37), sign[i % 3 % 2 + 1] * (i % 9 + 2.25)
  end },
  { "integer % float", ints, floats, function(i)
    return sign[i % 2 + 1] * (i * 13), sign[i % 3 % 2 + 1] * (i % 9 + 2.25)
  end },
  { "float % integer", floats, ints, function(i)
    return sign[i % 2 + 1] * (i * 0.37), sign[i % 3 % 2 + 1] * (i % 9 + 2)
  end },
  { "either % either", both, both, function(i)
    local a = sign[i % 2 + 1] * (i * 13)
    local b = sign[i % 3 % 2 + 1] * (i % 9 + 2)
    if i % 4 < 2 then a = a + 0.5 end
    if i % 5 < 2 then b = b + 0.25 end
    return a, b
  end },
}
for i = 1, #pairs_of do
  local name, left, right, ordinary = pairs_of[i][1], pairs_of[i][2], pairs_of[i][3], pairs_of[i][4]
  local xs, ys = operands(left, right, ordinary)
  -- A loop of its own for each table, so each is compiled for its operands.
  local loop = compile([[
    local set = ...
    return function(xs, ys, out, from, to)
      for i = from, to do
        set(i)
        out[i] = xs[i] % ys[i]
      end
    end
  ]])(function(i) at = i end)
  report(name, xs, ys, loop)
  report(name .. ", shared loop", xs, ys, remainders)
end

-- Divisors the compiler knows: a constant takes forms of its own (a power of
-- two, a multiply for the quotient, a float divisor of one sign).
local divisors = {
  "1", "-1", "2", "-2", "7", "-7", "8", "-8", "101", "-101", "4096",
  "2.5", "-2.5", "0.5", "8.0", "-8.0", "0.0", "-0.0", "(1/0)", "(-1/0)",
  "1e300", "9007199254740993", "9223372036854775807",
}
local dividends = {
  { "integers", ints, function(i) return sign[i % 2 + 1] * (i * 13) end },
  { "floats", floats, function(i) return sign[i % 2 + 1] * (i * 0.37) end },
  { "either", both, function(i)
    local a = sign[i % 2 + 1] * (i * 13)
    if i % 4 < 2 then a = a + 0.5 end
    return a
  end },
}
for d = 1, #divisors do
  for k = 1, #dividends do
    local name, pool, ordinary = dividends[k][1], dividends[k][2], dividends[k][3]
    local xs = {}
    for i = 1, WARM do xs[i] = ordinary(i) end
    for i = 1, #pool do xs[#xs + 1] = pool[i] end
    local loop = compile([[
      return function(xs, out, n)
        for i = 1, n do out[i] = xs[i] % ]] .. divisors[d] .. [[ end
      end
    ]])()
    local out = {}
    loop(xs, out, #xs)
    local sum, shown = 0, {}
    for i = 1, WARM do sum = sum + out[i] end
    for i = WARM + 1, #xs do shown[#shown + 1] = show(out[i]) end
    print(name .. " % " .. divisors[d], show(sum), table.concat(shown, " "))
  end
end
