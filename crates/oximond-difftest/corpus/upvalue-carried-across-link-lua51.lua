-- oximond-difftest repro profile=lua51 from=run-jit campaign (seed lua51
-- 11117411798829551595), reduced. A hot loop calls a counter closure (hg38),
-- whose upvalue n the trace carries as a loop parameter, and rebinds a table
-- the loop indexes. Under tracing hg38(0) was 39047342 instead of 203205552:
-- a root exit after a link from a side compiled apart wrote the link's loop
-- parameters back to the frame's registers but not to the upvalue cell, which
-- kept the value of the run's first entry. Fixed; kept as a regression.
local function show(v)
  local t = type(v)
  if t == "number" then if v ~= v then return "nnan" end if v == 0 then return "n0" end return "n" .. string.format("%.17g", v) end
end
local function out(tag, ...)
  local n = select("#", ...)
  local parts = {}
  for k = 1, n do parts[k] = show((select(k, ...))) end
  print(tag .. "#" .. n .. " " .. table.concat(parts, " "))
end
local function nz(x) if x == 0 then return 1 end return x end
local function ek(e)
end
local function array_index(v, len)
  local remainder = v % len
  return math.floor(remainder) + 1
end
local ClassB = setmetatable({}, {__index = ClassA})
local Proxy = setmetatable({}, {
})
local g1 = 0
local g3 = string.format("%5s|%-5s|", (string.gsub(" 3.5 ", "%z", "")), (string.gsub("10", "%z", "")))
local g4 = {g1, g1}
local g7 = ((-(9)) % 1000003)
local g8 = g1
local gen = coroutine.wrap(function()
end)
local envf = setfenv(function(x) return tostring(v) .. ":" .. string_number(x) end, {v = 7, tostring = tostring, string_number = string_number})
local function f26(p27, p28)
end
do -- kernel section 0
  local k32 = {1, (-10)}
  local function hf34(a, b) if a > b then return (a - b) % 1000003 end return (a * 3 + b) % 1000003 end
  local function hb36(d) hc35 = hc35 + d return hc35 end
  local function hm37() local n = 0 return function(d) n = n + d return n end end
  local hg38 = hm37()
  local hh39 = {}
  for i40 = 1, 35339 do
    hh39["k" .. ((true and g7 or 5) % 7)] = (hh39["k" .. (1284 % 7)] or 0) + 1
    g1 = ((g1 + ((unpack(g4)) or 0)) % 1000003)
    g7 = ((hf34(g7, g1)) % 1000003)
    g8 = ((hg38((g4[array_index((assert(g8)), #g4)] or 0)) + g8) % 1000003)
    for j41 = 1, 3 do
      if i40 == 10744 then
        g4 = {(k32[array_index(i40, #k32)] or 0), g1}
      end
      if i40 == 33479 then
        g4 = {8, 0, g1, j41, g8}
      end
    end
  end
  print(hg38(0), g1, g7, g8, hh39.k3, hh39.k0)
end