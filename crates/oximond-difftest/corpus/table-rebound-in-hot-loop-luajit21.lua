-- oximond-difftest repro profile=luajit21 from=run-jit campaign (seed luajit21
-- 2723), reduced. A hot loop rebinds a table (g4) at one iteration, then reads
-- it with unpack and writes it by a computed index. Under tracing the final k18
-- was 743859 instead of 500043: a root exit after a link from a side compiled
-- apart left the loop variables of the run's first entry, and the interpreter
-- ran those iterations again. Fixed; kept as a regression.
local function string_numbers(t)
end
local function show(v)
  local t = type(v)
  if t == "number" then if v ~= v then return "nnan" end return "n" .. string.format("%.17g", v) end
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
  local remainder = v % len
  return math.floor(remainder) + 1
end
local numeric_strings = {"10", "42", "7", " 7 ", "0x10", "2.5", "1e2", "-3", "0.5", "3."}
local Proxy = setmetatable({}, {
})
local g1 = (({(select("#", unpack(({(-5), Proxy.key3, 5, (#"0x1F")}), 2, 3))), 0xff, 10})[array_index(4, #({(select("#", unpack(({(-5), Proxy.key3, 5, (#"0x1F")}), 2, 3))), 0xff, 10}))] or 0)
local g3 = "hello"
local g4 = {select("#", select(2, (-8), g1, 3)), math.floor(g1 / nz(g1))}
G11 = select("#", select(2, g1, bit.band((array_index(g1, 321) - 21), 200), math.floor(g1 / nz(1924))))
local function f12(p13, p14, p15)
end
local k18 = (math.max(G11, G11) % nz(g1))
local k21 = {1, (-8), 8, g1}
local ck22 = 0
local function hb24(d) hc23 = hc23 + d return hc23 end
local hh27 = {}
for i28 = 19643, 1, -1 do
  if i28 == 2385 then
    g4 = {9007199254740992, g1, (0xff % nz(g1)), k18, i28}
  end
  hh27["k" .. (5 % 7)] = (hh27["k" .. ((((bit.lshift(57, (array_index(G11, 321) - 21) % 31) + 9007199254740992)) % 1000003) % 7)] or 0) + 1
  k18 = ((k18 + ((unpack(g4)) or 0)) % 1000003)
  g4[array_index(k18, #g4)] = ((g1) % 1000003)
  G11 = ((G11 + (k21[array_index(9, #k21)] or 0)) % 1000003)
  g1 = ((g1 + (assert((select(-1, i28, G11, (((G11 + k18)) % 1000003))), "message"))) % 1000003)
end
out("K3", ck22, k18, k19, k20, arr(k21))