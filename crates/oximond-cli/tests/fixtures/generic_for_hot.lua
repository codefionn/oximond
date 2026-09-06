local out = {}
local function P(...) local t = {} for i = 1, select('#', ...) do t[#t+1] = tostring((select(i, ...))) end out[#out+1] = table.concat(t, " ") end
local t = {}
for i = 1, 3000 do t[i] = i end
-- plain ipairs sum, repeated so the loop gets hot
local total = 0
for rep = 1, 20 do for i, v in ipairs(t) do total = total + v + i end end
P("sum", total)
-- ipairs with float elements
local f = {}
for i = 1, 2000 do f[i] = i + 0.5 end
local s = 0
for rep = 1, 20 do for _, v in ipairs(f) do s = s + v end end
P("float", s)
-- type change midway
local mixed = {}
for i = 1, 2000 do mixed[i] = i end
mixed[1500] = "x"
local ok, e = pcall(function() local z = 0 for rep = 1, 5 do for _, v in ipairs(mixed) do z = z + v end end return z end)
P("mixed", ok, (tostring(e):gsub("^[^:]*:%d+: ", "")))
-- hole in the middle
local holes = {}
for i = 1, 2000 do holes[i] = i end
holes[1000] = nil
local c = 0
for rep = 1, 20 do for _ in ipairs(holes) do c = c + 1 end end
P("holes", c)
-- nested loops
local grid = {}
for i = 1, 60 do grid[i] = {} for j = 1, 60 do grid[i][j] = i * j end end
local g = 0
for rep = 1, 10 do for _, row in ipairs(grid) do for _, v in ipairs(row) do g = g + v end end end
P("nested", g)
-- pairs on array and hash, hot
local h = {}
for i = 1, 1000 do h["k" .. i] = i end
local hs = 0
for rep = 1, 20 do for k, v in pairs(h) do hs = hs + v end end
P("pairs", hs)
local arr = 0
for rep = 1, 20 do for k, v in pairs(t) do arr = arr + k + v end end
P("pairs_arr", arr)
-- closure iterator hot
local function range(n) local i = 0 return function() i = i + 1 if i <= n then return i end end end
local rs = 0
for rep = 1, 20 do for i in range(1000) do rs = rs + i end end
P("range", rs)
-- stateless iterator hot
local function inc(n, i) if i < n then return i + 1 end end
local ss = 0
for rep = 1, 20 do for i in inc, 1000, 0 do ss = ss + i end end
P("stateless", ss)
-- closures capturing the loop var in a hot loop
local fs = {}
for rep = 1, 3 do for i, v in ipairs(t) do if i % 1000 == 0 then fs[#fs+1] = function() return i + v end end end end
local cs = 0
for _, fn in ipairs(fs) do cs = cs + fn() end
P("captured", #fs, cs)
-- break out of a hot loop and the exit path
local found
for rep = 1, 30 do for i, v in ipairs(t) do if v == 2500 then found = i break end end end
P("break", found)
-- loop variable assignment in hot loop
local assign = (loadstring or load)([[
  local t = ...
  local acc = 0
  for rep = 1, 20 do for i, v in ipairs(t) do i = i * 2 acc = acc + i end end
  return acc]])
P("assign", assign and assign(t) or "const loop variables")
-- loop with side-exit when table grows
local grow = {1, 2, 3}
local gc = 0
for rep = 1, 200 do for i, v in ipairs(grow) do gc = gc + v end grow[#grow + 1] = rep end
P("grow", gc)
-- ipairs on tables with metatables (__index)
local proxy = setmetatable({}, {__index = function(_, k) if k <= 100 then return k end end})
local ps = 0
for rep = 1, 50 do for i, v in ipairs(proxy) do ps = ps + v end end
P("proxy", ps)
-- strings in values
local words = {}
for i = 1, 500 do words[i] = "w" .. i end
local wl = 0
for rep = 1, 20 do for _, w in ipairs(words) do wl = wl + #w end end
P("strings", wl)
-- error inside hot loop body
local ok2, e2 = pcall(function() for rep = 1, 3 do for i, v in ipairs(t) do if i == 2900 and rep == 3 then error("stop " .. i) end end end end)
P("error", ok2, (tostring(e2):gsub("^[^:]*:%d+: ", "")))
-- return from inside the loop (function trace)
local function first_big(tbl, n) for i, v in ipairs(tbl) do if v > n then return i end end return -1 end
local fb = 0
for k = 1, 3000 do fb = fb + first_big(t, k % 50) end
P("return", fb)
print(table.concat(out, "\n"))
