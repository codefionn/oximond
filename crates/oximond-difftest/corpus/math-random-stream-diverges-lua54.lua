-- oximond-difftest repro profile=lua54 from=run-jit campaign (seed lua54 728),
-- reduced. A hot loop that calls math.random() once per iteration and rewrites
-- a table part way through drew further along the stream under tracing: the
-- last k26 was 75.53... instead of 20.55.... A root exit after a link from a
-- side compiled apart left the loop variables of the run's first entry, so the
-- interpreter ran those iterations, and their draws, again. Fixed; kept as a
-- regression.
local function string_number(v)
end
local function string_numbers(t)
end
local function show(v)
  local t = type(v)
  if t == "number" then if math.type(v) == "integer" then return "i" .. tostring(v) end if v ~= v then return "fnan" end return "f" .. string.format("%.17g", v) end
  return t
end
local function out(tag, ...)
  local n = select("#", ...)
  local parts = {}
  for k = 1, n do parts[k] = show((select(k, ...))) end
  print(tag .. "#" .. n .. " " .. table.concat(parts, " "))
end
local function arr(t)
end
local function rec(r)
end
local function nz(x) if x == 0 then return 1 end return x end
local function ek(e)
end
local function array_index(v, len)
  if len < 1 or v ~= v or v == math.huge or v == -math.huge then return 1 end
  local remainder = v % len
  return math.floor(remainder) + 1
end
local numeric_strings = {"10", "42", "7", " 7 ", "0x10", "2.5", "1e2", "-3", "0.5", "3."}
local function bit_operand(v)
  local i = math.tointeger(v)
  return math.floor(v % 1048576)
end
local function numeric_less(a, b)
end
math.randomseed(42)
local ClassA = {}
local g1 = (-7)
local g8 = string.format("%s %s %%", string_number(4), tostring(g7))
G9 = (-11)
local k22 = (bit_operand((array_index(G9, 321) - 21)) >> (bit_operand((array_index(g1, 321) - 21)) % 70))
local k25 = {G9, 11}
local function hf28(a, b) if a > b then return (a - b) % 1000003 end return (a * 3 + b) % 1000003 end
local hh29 = {}
for i30 = 35830, 1, -1 do
  if i30 == 10145 then
    k25 = {j31}
  end
  hh29["k" .. (3116 % 7)] = (hh29["k" .. ((i30 + (-0x80000000)) % 7)] or 0) + 1
  k25[array_index((math.abs(i30) % nz(10)), #k25)] = ((k25[array_index((~bit_operand(282)), #k25)] or 0) * math.abs(i30))
  k26 = (math.random() * 100)
  k22 = hf28(k22, ((table.unpack(k25)) or 0))
  k25[array_index((bit_operand(144) ~ bit_operand(170)), #k25)] = (select("#", table.unpack(k25)))
end
out("K2", ck27, k22, k23, k24, arr(k25), k26)
